import sys
import argparse

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log, loadImage

parser = argparse.ArgumentParser()

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

net = imageNet(model="model/resnet18.onnx", labels="model/labels.txt", input_blob="input_0", output_blob="output_0")

img = loadImage(args.input)

input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
font = cudaFont()

classID, confidence = net.Classify(img)

classLabel = net.GetClassLabel(classID)
confidence *= 100.0

print(f"clothing type: {confidence:05.2f}% {classLabel}")

font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}")
                         
output.Render(img)


