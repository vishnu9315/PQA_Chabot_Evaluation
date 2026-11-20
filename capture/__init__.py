"""Capture pipeline for the MeitY PQA chatbot (PQA_ragas/capture_new_chatbot.py).

Pure, network-free pieces (sse.py, dataset_io.py) are unit-tested without any live calls.
client.py is the only module that performs real HTTP requests, and only when invoked
through capture_new_chatbot.py - importing this package never makes a network call.
"""
