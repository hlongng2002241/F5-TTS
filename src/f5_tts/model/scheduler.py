class MelAttnAlphaScheduler:
    def __init__(self, start_value=0.0, end_value=1.0, num_steps=10000):
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps

    def get_alpha(self, current_step: int):
        if current_step >= self.num_steps:
            return self.end_value
        progress = current_step / self.num_steps
        return self.start_value + (self.end_value - self.start_value) * progress
