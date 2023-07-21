import sys
import argparse
import os

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log, loadImage

parser = argparse.ArgumentParser()

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

net = imageNet(model="model/resnet18.onnx", labels="model/labels.txt", input_blob="input_0", output_blob="output_0")

input = videoSource(args.input, argv=sys.argv)
font = cudaFont()

if not os.path.exists("outputs/dress"):
    os.makedirs("outputs/dress")
if not os.path.exists("outputs/shirt"):
    os.makedirs("outputs/shirt")
if not os.path.exists("outputs/shorts"):
    os.makedirs("outputs/shorts")
if not os.path.exists("outputs/shoes"):
    os.makedirs("outputs/shoes")
if not os.path.exists("outputs/pants"):
    os.makedirs("outputs/pants")

i = 0
while True:
    img = input.Capture()

    if img is None:
        continue
        
    classID, confidence = net.Classify(img)

    classLabel = net.GetClassLabel(classID)
    confidence *= 100.0

    print(f"clothing type: {confidence:05.2f}% {classLabel}")
    font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}")
                        
    output = videoOutput(f"outputs/{classLabel}/output_{i}.jpg", argv=sys.argv)
    i += 1
    output.Render(img)

    if not input.IsStreaming() or not output.IsStreaming():
        break


