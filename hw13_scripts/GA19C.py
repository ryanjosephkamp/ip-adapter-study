# GA19C.py CS6078 cheng 2026
# Chapter 9 of Hands-on GenAI
# Usage: python GA19C.py

import torch
from transformers import pipeline
pipe = pipeline("text-to-speech", model="suno/bark-small")
text = "Ladybugs have had important roles in culture and religion, being associated with luck, love, fertility and prophecy. "
output = pipe(text)

print('done')
import scipy
scipy.io.wavfile.write("GA19Cout1.wav", rate=output["sampling_rate"], data=output["audio"])

'''
from IPython.display import Audio
Audio(output["audio"].squeeze(), rate=output["sampling_rate"])
'''
