from typing import List
import numpy as np

class VariableNode:
    def __init__(self):
        self.eta = 0
        self.Lambda = 0

        self.links = []

        self.messages = {} #Dictionary of factor -> (eta, Lambda)

    def belief_update(self):
        self.eta = 0
        self.Lambda = 0
        for (eta, Lambda) in self.messages.values():
            self.eta += eta
            self.Lambda += Lambda
    
    def send_message(self, factor: FactorNode):
        m_eta = 0
        m_Lambda = 0
        for (link, (eta, Lambda)) in self.messages.items():
            if link != factor:
                m_eta += eta
                m_Lambda += Lambda
        
        outlink.messages[self] = (m_eta, m_Lambda)

    def send_messages(self):
        for link in self.links:
            self.send_message(link)

class FactorNode:
    def __init__(self, variables: List[VariableNode], eta: np.ndarray, Lambda: np.ndarray):
        self.eta = eta
        self.Lambda = Lambda

        for var in variables:
            var.links.append(self)
        self.links = variables

        self.messages = {}

    def send_message(self, var: VariableNode):
        eta_prime = self.eta.copy()
        Lambda_prime = self.Lambda.copy()
        
        for (i, link) in enumerate(self.links):
            if link != var:
                (_, (eta, Lambda)) = self.messages[link]
                eta_prime[i] += eta
                Lambda_prime[i, i] += Lambda
        
        # Marginalise to get a scalar message
        var_idx = self.links.index(var)
        eta_prime[0], eta_prime[var_idx] = eta_prime[var_idx], eta_prime[0]
        Lambda_prime[0, :], Lambda_prime[var_idx, :] = Lambda_prime[var_idx, :], Lambda_prime[0, :]
        Lambda_prime[:, 0], Lambda_prime[:, var_idx] = Lambda_prime[:, var_idx], Lambda_prime[:, 0]

        eta_a = eta_prime[0]
        eta_b = eta_prime[1:]
        Lambda_a_a = Lambda_prime[var_idx, var_idx]
        Lambda_a_b = Lambda_prime[var_idx, 1:]
        Lambda_b_a = Lambda_prime[1:, var_idx]
        Lambda_b_b = Lambda_prime[1:, 1:]

        partial_dot = np.dot(Lambda_a_b, Lambda_b_b.I)
        m_eta = eta_a - np.dot(partial_dot, eta_b)
        m_Lambda = Lambda_a_a - np.dot(partial_dot, Lambda_b_a)

    def send_messages(self):
        for link in self.links:
            self.send_message(link)

