class ClassicalController:
    def __init__(self, k_p, k_i, k_d):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.error_history = []

    def calc_proportinal(self):
        return self.k_p * self.error_history[-1]
    
    def calc_integral(self):
        return self.k_i * sum(self.error_history)
    
    def calc_derivative(self):
        return self.k_d * ( self.error_history[-1] - self.error_history[-2] )
    
    def get_control_signal(self):
        return self.calc_proportinal + self.calc_integral() + self.calc_derivative()

    def step(self, error):
        self.error_history.append(error)
        return self.get_control_signal()

