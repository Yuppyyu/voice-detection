from utils.responses import responses


class Bot:
    def __init__(self):
        self.name = ("Friday")

    def get_response(self, user_input):
        # Make the input lowercase to make it easier to match.
        user_input = user_input.lower()
        # Check if we have a prefeinded responce for that input
        return responses.get(user_input, f"I'm not sure how to responded to '{user_input}'.")
