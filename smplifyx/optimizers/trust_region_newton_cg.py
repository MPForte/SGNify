# -*- coding: utf-8 -*-

# @author Vasileios Choutas
# Contact: vassilis.choutas@tuebingen.mpg.de


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from typing import NewType, List, Tuple
import warnings

import torch
import torch.optim as optim
import torch.autograd as autograd

import math

Tensor = NewType('Tensor', torch.Tensor)


class TrustRegionNewtonCG(optim.Optimizer):
    # torch.autograd.set_detect_anomaly(True)
    def __init__(self, params: List[Tensor],
                 max_trust_radius: float = 1000,
                 initial_trust_radius: float = 0.05,
                 eta: float = 0.15,
                 gtol: float = 1e-05,
                 **kwargs) -> None:
        ''' Trust Region Newton Conjugate Gradient

            Uses the Conjugate Gradient Algorithm to find the solution of the
            trust region sub-problem

            For more details see chapter 7.2 of
            "Numerical Optimization, Nocedal and Wright"

            Arguments:
                params (iterable): A list or iterable of tensors that will be
                    optimized
                max_trust_radius: float
                    The maximum value for the trust radius
                initial_trust_radius: float
                    The initial value for the trust region
                eta: float
                    Minimum improvement ration for accepting a step
                gtol: float
                    Gradient tolerance for stopping the optimization
        '''
        defaults = dict()

        super(TrustRegionNewtonCG, self).__init__(params, defaults)

        self.steps = 0
        self.max_trust_radius = max_trust_radius
        self.initial_trust_radius = initial_trust_radius
        self.eta = eta
        self.gtol = gtol
        self._params = self.param_groups[0]['params']

    @torch.enable_grad()
    def _compute_hessian_vector_product(
            self,
            gradient: Tensor,
            p: Tensor) -> Tensor:

        hess_vp = autograd.grad(
            torch.sum(gradient * p, dim=-1), self._params,
            only_inputs=True, retain_graph=True, allow_unused=True)
        return torch.cat(
            [torch.flatten(vp) for vp in hess_vp], dim=-1)
        #  hess_vp = torch.cat(
            #  [torch.flatten(vp) for vp in hess_vp], dim=-1)
        #  return torch.flatten(hess_vp)

    def _gather_flat_grad(self) -> Tensor:
        ''' Concatenates all gradients into a single gradient vector
        '''
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        output = torch.cat(views, 0)
        return output

    @torch.no_grad()
    def _improvement_ratio(self, p, start_loss, gradient, closure):
        ''' Calculates the ratio of the actual to the expected improvement

            Arguments:
                p (torch.tensor): The update vector for the parameters
                start_loss (torch.tensor): The value of the loss function
                    before applying the optimization step
                gradient (torch.tensor): The flattened gradient vector of the
                    parameters
                closure (callable): The function that evaluates the loss for
                    the current values of the parameters
            Returns:
                The ratio of the actual improvement of the loss to the expected
                improvement, as predicted by the local quadratic model
        '''

        # Apply the update on the parameter to calculate the loss on the new
        # point
        hess_vp = self._compute_hessian_vector_product(gradient, p)

        # Apply the update of the parameter vectors.
        # Use a torch.no_grad() context since we are updating the parameters in
        # place
        with torch.no_grad():
            start_idx = 0
            for param in self._params:
                num_els = param.numel()
                curr_upd = p[start_idx:start_idx + num_els]
                param.data.add_(curr_upd.view_as(param))
                start_idx += num_els

        # No need to backpropagate since we only need the value of the loss at
        # the new point to find the ratio of the actual and the expected
        # improvement
        new_loss = closure(backward=False)
        # The numerator represents the actual loss decrease
        numerator = start_loss - new_loss

        new_quad_val = self._quad_model(p, start_loss, gradient, hess_vp)

        # The denominator
        denominator = start_loss - new_quad_val

        # TODO: Convert to epsilon, print warning
        ratio = numerator / (denominator + 1e-20)
        return ratio

    @torch.no_grad()
    def _quad_model(
            self,
            p: Tensor,
            loss: float,
            gradient: Tensor,
            hess_vp: Tensor) -> float:
        ''' Returns the value of the local quadratic approximation
        '''
        return (loss + torch.flatten(gradient * p).sum(dim=-1) +
                0.5 * torch.flatten(hess_vp * p).sum(dim=-1))

    @torch.no_grad()
    def calc_boundaries(
            self,
            iterate: Tensor,
            direction: Tensor,
            trust_radius: float) -> Tuple[Tensor, Tensor]:
        ''' Calculates the offset to the boundaries of the trust region
        '''

        a = torch.sum(direction ** 2, dim=-1)
        b = 2 * torch.sum(direction * iterate, dim=-1)
        c = torch.sum(iterate ** 2, dim=-1) - trust_radius ** 2
        sqrt_discriminant = torch.sqrt(b * b - 4 * a * c)
        ta = (-b + sqrt_discriminant) / (2 * a)
        tb = (-b - sqrt_discriminant) / (2 * a)
        if ta.item() < tb.item():
            return [ta, tb]
        else:
            return [tb, ta]

    @torch.no_grad()
    def _solve_trust_reg_subproblem(
            self,
            loss: float,
            flat_grad: Tensor,
            trust_radius: float) -> Tuple[Tensor, bool]:
        ''' Solves the quadratic subproblem in the trust region
        '''

        # The iterate vector that contains the increment from the starting
        # point
        iterate = torch.zeros_like(flat_grad, requires_grad=False)

        # The residual of the CG algorithm
        residual = flat_grad.detach()
        # The first direction of descent
        direction = -residual

        jac_mag = torch.norm(flat_grad).item()
        # Tolerance define in Nocedal & Wright in chapter 7.1
        tolerance = min(0.5, math.sqrt(jac_mag)) * jac_mag

        # If the magnitude of the gradients is smaller than the tolerance then
        # exit
        if jac_mag <= tolerance:
            return iterate, False

        # Iterate to solve the subproblem
        while True:
            # Calculate the Hessian-Vector product
            #  start = time.time()
            hessian_vec_prod = self._compute_hessian_vector_product(
                flat_grad, direction
            )
            #  torch.cuda.synchronize()
            #  print('Hessian Vector Product', time.time() - start)

            # This term is equal to p^T * H * p
            #  start = time.time()
            hevp_dot_prod = torch.sum(hessian_vec_prod * direction)
            #  print('p^T H p', time.time() - start)

            # If non-positive curvature
            if hevp_dot_prod.item() <= 0:
                # Find boundaries and select minimum
                #  start = time.time()
                ta, tb = self.calc_boundaries(iterate, direction, trust_radius)
                pa = iterate + ta * direction
                pb = iterate + tb * direction

                # Calculate the point on the boundary with the smallest value
                bound1_val = self._quad_model(pa, loss, flat_grad,
                                              hessian_vec_prod)
                bound2_val = self._quad_model(pb, loss, flat_grad,
                                              hessian_vec_prod)
                #  torch.cuda.synchronize()
                #  print('First if', time.time() - start)
                #  print()
                if bound1_val.item() < bound2_val.item():
                    return pa, True
                else:
                    return pb, True

            # The squared euclidean norm of the residual needed for the CG
            # update
            #  start = time.time()
            residual_sq_norm = torch.sum(residual * residual, dim=-1)

            # Compute the step size for the CG algorithm
            cg_step_size = residual_sq_norm / hevp_dot_prod

            # Update the point
            next_iterate = iterate + cg_step_size * direction

            iterate_norm = torch.norm(next_iterate, dim=-1)
            #  torch.cuda.synchronize()
            #  print('CG Updates', time.time() - start)

            # If the point is outside of the trust region project it on the
            # border and return
            if iterate_norm.item() >= trust_radius:
                #  start = time.time()
                ta, tb = self.calc_boundaries(iterate, direction, trust_radius)
                p_boundary = iterate + tb * direction

                #  torch.cuda.synchronize()
                #  print('Second if', time.time() - start)
                #  print()
                return p_boundary, True

            #  start = time.time()
            # Update the residual
            next_residual = residual + cg_step_size * hessian_vec_prod
            #  torch.cuda.synchronize()
            #  print('Residual update', time.time() - start)
            # If the residual is small enough, exit
            if torch.norm(next_residual, dim=-1).item() < tolerance:
                #  print()
                return next_iterate, False

            #  start = time.time()
            beta = torch.sum(next_residual ** 2, dim=-1) / residual_sq_norm
            # Compute the new search direction
            direction = (-next_residual + beta * direction).squeeze()
            if torch.isnan(direction).sum() > 0:
                raise RuntimeError

            iterate = next_iterate
            residual = next_residual
            #  torch.cuda.synchronize()
            #  print('Replacing vectors', time.time() - start)
            #  print(trust_radius)
            #  print()

    def step(self, closure) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            starting_loss = closure(backward=True)

        flat_grad = self._gather_flat_grad()

        state = self.state
        if len(state) == 0:
            state['trust_radius'] = torch.full([1],
                                               self.initial_trust_radius,
                                               dtype=flat_grad.dtype,
                                               device=flat_grad.device)
        trust_radius = state['trust_radius']

        param_step, hit_boundary = self._solve_trust_reg_subproblem(
            starting_loss, flat_grad, trust_radius)
        self.param_step = param_step

        if torch.norm(param_step).item() <= self.gtol:
            return starting_loss

        improvement_ratio = self._improvement_ratio(
            param_step, starting_loss, flat_grad, closure)

        if improvement_ratio.item() < 0.25:
            trust_radius.mul_(0.25)
        else:
            if improvement_ratio.item() > 0.75 and hit_boundary:
                trust_radius.mul_(2).clamp_(0.0, self.max_trust_radius)

        if improvement_ratio.item() <= self.eta:
            # If the improvement is not sufficient, then undo the update
            start_idx = 0
            for param in self._params:
                num_els = param.numel()
                curr_upd = param_step[start_idx:start_idx + num_els]
                param.data.add_(-curr_upd.view_as(param))
                start_idx += num_els

        self.steps += 1
        return starting_loss
