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

    def get_dist(self):
        # return self.Lambda * self.eta, 1/self.Lambda
        return self.eta, self.Lambda
    
    def calculate_message(self, factor):
        m_eta = 0
        m_Lambda = 0
        for (link, (eta, Lambda)) in self.messages.items():
            if link != factor:
                m_eta += eta
                m_Lambda += Lambda
        return m_eta, m_Lambda

    def send_messages(self):
        for link in self.links:
            m_eta, m_Lambda = self.calculate_message(link)
            link.messages[self] = (m_eta, m_Lambda)

class FactorNode:
    def __init__(self, variables: List[VariableNode], mean: np.ndarray, cov: np.ndarray):
        self.update(mean, cov)

        for var in variables:
            var.links.append(self)
        self.links = variables

        self.messages = {}

    def set_eta_lambda(self, eta, Lambda):
        self.eta = eta
        self.Lambda = Lambda

    def calculate_message(self, var: VariableNode):
        eta_prime = self.eta.copy()
        Lambda_prime = self.Lambda.copy()
        
        for (i, link) in enumerate(self.links):
            if link != var:
                (eta, Lambda) = self.messages.setdefault(link, (0, 0))
                eta_prime[i] += eta
                Lambda_prime[i, i] += Lambda

        # Marginalise to get a scalar message
        var_idx = self.links.index(var)
        eta_prime[0], eta_prime[var_idx] = eta_prime[var_idx], eta_prime[0]
        Lambda_prime[[0, var_idx]] = Lambda_prime[[var_idx, 0]]
        Lambda_prime[:, [0, var_idx]] = Lambda_prime[:, [var_idx, 0]]

        eta_a = eta_prime[0]
        eta_b = eta_prime[1:]
        Lambda_a_a = Lambda_prime[0, 0]
        Lambda_a_b = Lambda_prime[0, 1:]
        Lambda_b_a = Lambda_prime[1:, 0]
        Lambda_b_b = Lambda_prime[1:, 1:]

        partial_dot = np.dot(Lambda_a_b, np.linalg.inv(Lambda_b_b))
        m_eta = eta_a - np.dot(partial_dot, eta_b)
        m_Lambda = Lambda_a_a - np.dot(partial_dot, Lambda_b_a)

        # print(eta_a, eta_b, Lambda_a_a, Lambda_a_b, Lambda_b_b, Lambda_b_a, m_eta, m_Lambda)

        return m_eta, m_Lambda

    def send_messages(self):
        for link in self.links:
            eta, Lambda = self.calculate_message(link)
            link.messages[self] = (eta, Lambda)

    def update(self, mean: np.ndarray, cov: np.ndarray):
        self.Lambda = np.linalg.inv(cov)
        self.eta = np.dot(self.Lambda, mean)