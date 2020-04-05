
class VariableTracker(object):
    """This class implements several methods to keep track of intermediate
    variables.
    This is useful for eg. visualizing or retrieving gradients wrt. inputs
    later on"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tracker = {}

    def track(self, variable, name, retain_grad=False):
        """Adds variable to the tracker
        Specify `retain_grad=True` to retrieve the gradient later."""
        if retain_grad:
            variable.retain_grad()
        self.tracker[name] = variable

    def __getitem__(self, name):
        return self.tracker[name]
