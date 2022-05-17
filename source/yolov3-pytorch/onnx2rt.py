import onnx
import onnxruntime
import argparse
import sys, os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw, Image
import matplotlib.pyplot as plt

from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
import common
#from downloader import getFilePath

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def parse_args():
    parser = argparse.ArgumentParser(description="onnx_inference")
    parser.add_argument("--mode", type=str, default=None, help="onnx2rt or rtinfer")
    parser.add_argument('--model', type=str, help="model path",
                        default=None, dest='model')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

#Save tensorrt
def onnx2trt(onnx_model, trt_model):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30)
    print(1<<30)
    #builder.max_workspace_size = (1 << 30)
    print("FP16 : " ,builder.platform_has_fast_fp16)
    
    with open(onnx_model, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
                
    plan = builder.build_serialized_network(network,config)
    
    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(plan)

    #engine = builder.build_cuda_engine(network)
    buf = engine.serialize()
    
    with open(trt_model, 'wb') as f:
        f.write(buf)

def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color="blue"):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    show_img = Image.fromarray(image_raw)

    draw = ImageDraw.Draw(show_img)

    img_raw_w, img_raw_h = show_img.size
    #print(bboxes, confidences, categories)
    if bboxes is not None:
        #bboxes = bboxes * [img_raw_w, img_raw_h, img_raw_w, img_raw_h]

        for box, score, category in zip(bboxes, confidences, categories):
            x_coord, y_coord, width, height = box
            left = max(0, np.floor(x_coord - width / 2 + 0.5).astype(int))
            top = max(0, np.floor(y_coord - height / 2 + 0.5).astype(int))
            right = min(img_raw_w, np.floor(x_coord + width / 2 + 0.5).astype(int))
            bottom = min(img_raw_h, np.floor(y_coord + height / 2 + 0.5).astype(int))

            draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
            draw.text((left, top - 12), "{0} {1:.2f}".format(all_categories[category], score), fill=bbox_color)
    plt.imshow(show_img)
    plt.show()

    return show_img


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 608, 608]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def run_trt(onnx_model):
    """Create a TensorRT engine for ONNX-based YOLOv3 and run inference."""

    # Try to load a previously generated YOLOv3 network graph in ONNX format:
    onnx_file_path = onnx_model
    engine_file_path = "yolov3.trt"
    # Download a dog image and save it to the following file path:
    input_image_path = "C:\\data\\kitti_dataset\\kitti_yolo\\eval\\Images\\006781.png"
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (608, 608)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(input_image_path)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.shape[0:2] #image_raw.shape[0:2]
    
    np.save("trt_input.npy",image)

    # Output shapes expected by the post-processor
    #output_shapes = [(1, 39, 20, 15), (1, 39, 40, 30), (1, 39, 80, 60)]
    output_shapes = [(1,22743,13)]
    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print("Running inference on image {}...".format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        print(trt_outputs)
    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    
    #trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
    trt_outputs = trt_outputs[0].reshape(output_shapes[0])
    print(len(trt_outputs), trt_outputs[0].shape)
    np.save("trt_output_6781.npy", trt_outputs[0])
    
    # divide outputs to 3 layers
    out_dims = [19 * 19 * 3, 38 * 38 * 3, 76 * 76 * 3]
    out_wh = [[19,19],[38,38],[76,76]]
    trt_outputs_div = [trt_outputs[0][:out_dims[0], :],
                       trt_outputs[0][out_dims[0]:out_dims[0] + out_dims[1], :],
                       trt_outputs[0][out_dims[0] + out_dims[1]:, :]]
    
    for i, tod in enumerate(trt_outputs_div):
        print(out_dims[i])
        print(tod.shape)

    postprocessor_args = {
        "yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],  # A list of 3 three-dimensional tuples for the YOLO masks
        "yolo_anchors": [
            (15,36),
            (30,49),
            (19,115),
            (50,71),
            (76,106),  # A list of 9 two-dimensional tuples for the YOLO anchors
            (43,212),
            (115,145),
            (129,230),
            (194,280),
        ],
        "obj_threshold": 0.4,  # Threshold for object coverage, float value between 0 and 1
        "nms_threshold": 0.2,  # Threshold for non-max suppression algorithm, float value between 0 and 1
        "yolo_input_resolution": input_resolution_yolov3_HW,
    }

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    boxes, classes, scores = postprocessor.process(trt_outputs_div, out_wh, (shape_orig_WH))
    # Draw the bounding boxes onto the original input image and save it as a PNG file
    print(boxes, classes, scores)
    obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
    output_image_path = "kitti_img.png"
    obj_detected_img.save(output_image_path, "PNG")
    print("Saved image with bounding boxes of detected objects to {}.".format(output_image_path))


def main():
    print("main")
    args = parse_args()
    onnx_model = args.model
    #trt_model = onnx_model.replace("onnx","plan")
    
    run_trt(onnx_model = onnx_model)

    # if args.mode == "onnx2rt":
    #     onnx2trt(onnx_model=onnx_model, trt_model=trt_model)
    # elif args.mode == "rtinfer":
    #     run_trt()
    
    # model = onnx.load(args.model)
    
    # x = np.randn([1,3,608,608])
    
    # print(onnx.checker.check_model(model))
    

    # ort_session = onnxruntime.InferenceSession(args.model)

    # # ONNX 런타임에서 계산된 결과값
    # ort_inputs = {ort_session.get_inputs()[0].name: x}
    
    # ort_outs = ort_session.run(None, ort_inputs)

    # print("out : ", ort_outs)

if __name__ == "__main__":
    args = parse_args()
    main()