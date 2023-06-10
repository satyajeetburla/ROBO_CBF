import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.flatten_util import ravel_pytree
from functools import partial
from jax import grad, vmap


class CBFLoss:

    def __init__(self, hparams, network, dynamics, alpha, dual_vars, T_x):
        self._hparams = hparams
        self._network = network
        self._dynamics = dynamics
        self._alpha = alpha
        self._dual_vars = dual_vars
        self._norm_T_x = 1.0 #jnp.linalg.norm(T_x)
        self.delta_x =0.5 # Change the vause of delta_x here
        print("delta_x:", self.delta_x )
        print("value of Lip", self._hparams.lip_const_a,self._hparams.lip_const_b)

    @partial(jax.jit, static_argnums=0)
    def cbf_term(self, params, states, disturbances, inputs, lip_const_a, lip_const_b):

        def cbf_term_indiv(data):

            x, d, u, lip_a, lip_b = data[:4], data[4], data[5], data[6], data[7]

            scalar_network = lambda x_ : jnp.sum(self._network.apply(params, x_))
            dh = jax.grad(scalar_network)(x)
            

            term1 = jnp.dot(dh, self._dynamics.f(x, d) + self._dynamics.g(x) * u)
            term2 = self._alpha(scalar_network(x))
            
            
            cbf_term_ = term1 + term2
            print("cbf_shape",cbf_term_.shape, lip_a.shape )

            if self._hparams.robust is True:
                multiplier = self._norm_T_x * (self._hparams.delta_f + self._hparams.delta_g * jnp.linalg.norm(u))
                cbf_term_ -= jnp.linalg.norm(dh) * multiplier

            if self._hparams.use_lip_output_term is True:
                # cbf_term_ -= self._hparams.lip_const_a + self._hparams.lip_const_b * jnp.linalg.norm(u)
                #print("lip:",  self._hparams.lip_const_a,self._hparams.lip_const_b)
              
                cbf_term_ -= ((lip_a +lip_b * jnp.linalg.norm(u))*(self.delta_x))
            return cbf_term_

        full_data = jnp.hstack((states, disturbances, inputs, lip_const_a, lip_const_b))
        return jax.vmap(cbf_term_indiv, in_axes=(0))(full_data)
      
      
    def B_terms(self, params, states, disturbances, inputs):
        def B_ind(data):
            x, d, u = data[:4], data[4], data[5]
            scalar_network = lambda x_ : jnp.sum(self._network.apply(params, x_))
            
            dh = jax.grad(scalar_network)(x)
            def bval1(dh, x, d, u): 
              B1 = jnp.dot(dh, self._dynamics.f(x, d)) + self._alpha(scalar_network(x))
              if self._hparams.robust is True:
                  B1 -= jnp.linalg.norm(dh) *  (self._norm_T_x * self._hparams.delta_f)
              return B1
            def bval2(dh, x, d, u): 
              B2 = jnp.dot(dh, self._dynamics.g(x))
              return B2
            def bval3(dh, x, d, u): 
              B3 = 0
              if self._hparams.robust is True:
                  B3 = -jnp.linalg.norm(dh) *  (self._norm_T_x * self._hparams.delta_g)
              return B3
              
            B1_lambda = lambda x:jnp.sum(bval1(dh, x, d, u))
            B2_lambda = lambda x:jnp.sum(bval2(dh, x, d, u))
            B3_lambda = lambda x:jnp.sum(bval3(dh, x, d, u))
            
            B1_dx = jax.grad(B1_lambda)(x)
            B2_dx = jax.grad(B2_lambda)(x)
            B3_dx = jax.grad(B3_lambda)(x)
            return B1_dx,B2_dx,B3_dx  
        full_data = jnp.hstack((states, disturbances, inputs))
        B1_dx_data, B2_dx_data, B3_dx_data = jax.vmap(B_ind, in_axes=(0))(full_data)
        return B1_dx_data, B2_dx_data, B3_dx_data

    # @partial(jax.jit, static_argnums=0)
    def loss_fn(self, params, data_dict):
        loss, _, _, _ = self.__loss(params, data_dict)
        return loss

    def constraints_fn(self, params, data_dict):
        _, constraints, _, _ = self.__loss(params, data_dict)
        return {key: float(value) for (key, value) in constraints.items()}
    def diffs_fn(self, params, data_dict):
        _, _, diffs, losses = self.__loss(params, data_dict)

        return diffs

    def all_losses_fn(self, params, data_dict):
        _, _, _, losses = self.__loss(params, data_dict)
        return {key: float(value) for (key, value) in losses.items()}

    @staticmethod
    def loss_with_dual_var(dual_var, diff):
        if dual_var.shape == ():
            return dual_var * jnp.sum(jnn.relu(diff))
        return jnp.dot(dual_var, jnn.relu(diff))

    @staticmethod
    def const_satisfaction(const):
        frac_incorrect = jnp.sum(jnp.heaviside(const, 0)) / const.shape[0]
        return (1.0 - frac_incorrect) * 100.

    # @partial(jax.jit, static_argnums=0)
    def __loss(self, params, data_dict):

        consts = dict.fromkeys(['safe', 'unsafe', 'dyn'])
        diffs = dict.fromkeys(['safe', 'unsafe', 'dyn'])
        losses = dict.fromkeys(['safe', 'unsafe', 'dyn', 'param'])

        # h(x_safe) >= \gamma_safe <=> \gamma_safe - h(x_safe) <= 0
        safe_output = self._network.apply(params, data_dict['safe'])
        diffs['safe'] = self._hparams.gamma_safe - safe_output
        losses['safe'] = self.loss_with_dual_var(self._dual_vars['safe'], diffs['safe'])
        consts['safe'] = self.const_satisfaction(diffs['safe'])

        # h(x_unsafe) <= -\gamma_unsafe <=> \gamma_unsafe + h(x_unsafe) <= 0
        unsafe_output = self._network.apply(params, data_dict['unsafe'])
        diffs['unsafe'] = self._hparams.gamma_unsafe + unsafe_output
        losses['unsafe'] = self.loss_with_dual_var(self._dual_vars['unsafe'], diffs['unsafe'])
        consts['unsafe'] = self.const_satisfaction(diffs['unsafe'])

        # q(u, x) >= \gamma_dyn <=> \gamma_dyn - q(u, x) <= 0

        
        #Lip Calculation
        B1_dx_data, B2_dx_data, B3_dx_data = self.B_terms(params, data_dict['all'], data_dict['all_dists'], data_dict['all_inputs'])
#         print("B1_dx_data",B1_dx_data)
#         print("B3_dx_data",B3_dx_data)
#         print("b1shape",B1_dx_data.shape)
#         print("b3shape",B3_dx_data.shape)
#       print("b3shape",B3_dx_data.shape)
        norm_array_1 = jnp.linalg.norm(B1_dx_data, axis=1, keepdims=True)
#         print("norm_shape",norm_array_1.shape)
        #max_value_1 = jnp.max(norm_array_1)
#         print("B1 val max",max_value_1)
        norm_array_2 = jnp.linalg.norm(B2_dx_data, axis=1, keepdims=True)
#         print("norm_shape",norm_array_2.shape)
        #max_value_2 = jnp.max(norm_array_2)
#         print("B2 val max",max_value_2)        
        norm_array_3 = jnp.linalg.norm(B3_dx_data, axis=1, keepdims=True)
#         print("norm_shape",norm_array_3.shape)
        #max_value_3 = jnp.max(norm_array_3)
#         print("B3 val max",max_value_3)
        
#        self._hparams.lip_const_a = max_value_1
#        self._hparams.lip_const_b = max_value_2 + max_value_3
        print(data_dict['all'].shape)
        print(norm_array_1.shape)
        lip_const_a = norm_array_1
        lip_const_b = norm_array_2 + norm_array_3
        #lip_const_b = norm_array_2
        
        cbf_output = self.cbf_term(params, data_dict['all'], data_dict['all_dists'], data_dict['all_inputs'],lip_const_a, lip_const_b)
        diffs['dyn'] = self._hparams.gamma_dyn - cbf_output
        losses['dyn'] = self.loss_with_dual_var(self._dual_vars['dyn'], diffs['dyn'])
        consts['dyn'] = self.const_satisfaction(diffs['dyn'])
        
        # Penalize large values of the derivative
        def grad_indiv(x):
            scalar_network = lambda x_ : jnp.sum(self._network.apply(params, x_))
            return jax.grad(scalar_network)(x)
        grad_all_output = jax.vmap(grad_indiv, in_axes=(0))(data_dict['all'])
        losses['grad'] = self._hparams.lambda_grad * jnp.sum(jnp.square(grad_all_output))

        # Penalize large parameter values
        losses['param'] = self._hparams.lambda_param * jnp.sum(jnp.square(ravel_pytree(params)[0]))

# loss = safe_loss + unsafe_loss + dyn_loss + dh_loss + param_loss
        loss = losses['safe'] + losses['unsafe'] + losses['dyn'] + losses['param'] + losses['grad']

        return jnp.reshape(loss, ()), consts, diffs, losses
