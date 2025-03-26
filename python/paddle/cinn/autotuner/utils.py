import json

def candidate_join(name, candidate):
    return [{'name': a, 'value': b} for a, b in zip(name, candidate)]