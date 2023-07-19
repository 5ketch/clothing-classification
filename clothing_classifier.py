#!/usr/bin/env python3

import sys
import argparse

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

parser = argparse.ArgumentParser()

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

net = imageNet(model="model/resnet18.onnx", labels="model/labels.txt", input_blob="input_0", output_blob="output_0")

input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

while True:
    img = input.Capture()

    if img is None:
        continue  

    predictions = net.Classify(img, topK=args.topK)

    for n, (classID, confidence) in enumerate(predictions):
        classLabel = net.GetClassLabel(classID)
        confidence *= 100.0

        print(f"clothing type: {confidence:05.2f}% {classLabel}")
                         
    output.Render(img)

    if not input.IsStreaming() or not output.IsStreaming():
        break
