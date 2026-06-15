import argparse
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


LAYERS = {
    "conv1": (195, 128, False, False),
    "conv2": (384, 384, False, False),
    "gru1_input": (384, 1152, True, False),
    "gru1_recurrent": (384, 1152, True, True),
    "gru2_input": (384, 1152, True, False),
    "gru2_recurrent": (384, 1152, True, True),
    "gru3_input": (384, 1152, True, False),
    "gru3_recurrent": (384, 1152, True, True),
    "dense_out": (1536, 32, False, False),
    "vad_dense": (1536, 1, False, False),
}


def extract_array(text, name, dtype):
    pattern = re.compile(
        rf"static const (?:float|int|opus_int8) {re.escape(name)}\[[^\]]*\] = \{{(.*?)\}};",
        re.S,
    )
    match = pattern.search(text)
    if not match:
        raise KeyError(f"array not found: {name}")
    body = match.group(1).replace("f", "")
    return np.fromstring(body, sep=",", dtype=dtype)


def dense_matrix(text, name, nb_inputs, nb_outputs, sparse, recurrent):
    bias = extract_array(text, f"{name}_bias", np.float32)
    if sparse:
        packed = extract_array(text, f"{name}_weights_float", np.float32)
        idx = extract_array(text, f"{name}_weights_idx", np.int64)
        weights = np.zeros((nb_inputs, nb_outputs), dtype=np.float32)
        idx_pos = 0
        w_pos = 0
        for out0 in range(0, nb_outputs, 8):
            nb_blocks = int(idx[idx_pos])
            idx_pos += 1
            for _ in range(nb_blocks):
                in0 = int(idx[idx_pos])
                idx_pos += 1
                block = packed[w_pos : w_pos + 32]
                w_pos += 32
                for c in range(4):
                    for r in range(8):
                        weights[in0 + c, out0 + r] = block[c * 8 + r]
        if w_pos != packed.size:
            raise ValueError(f"{name}: used {w_pos} packed weights, have {packed.size}")
    else:
        raw = extract_array(text, f"{name}_weights_float", np.float32)
        weights = raw.reshape(nb_inputs, nb_outputs)
    if recurrent:
        diag = extract_array(text, f"{name}_weights_diag", np.float32)
        hidden = nb_inputs
        for i in range(hidden):
            weights[i, i] += diag[i]
            weights[i, hidden + i] += diag[hidden + i]
            weights[i, 2 * hidden + i] += diag[2 * hidden + i]
    return weights.astype(np.float32), bias.astype(np.float32)


def remove_unused_opset_imports(model):
    used_domains = {node.domain for node in model.graph.node}
    del model.opset_import[:]
    for domain, version in [("", 13)]:
        if domain in used_domains or domain == "":
            model.opset_import.append(helper.make_operatorsetid(domain, version))


def optimize_model(input_path, output_path):
    code = """
import sys
import onnxruntime as ort

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
session_options.optimized_model_filepath = sys.argv[2]
ort.InferenceSession(sys.argv[1], session_options, providers=["CPUExecutionProvider"])
"""
    try:
        subprocess.run([sys.executable, "-c", code, str(input_path), str(output_path)], check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("ONNX Runtime failed to optimize the model") from exc

    model = onnx.load(output_path)
    remove_unused_opset_imports(model)
    # Windows.AI.MachineLearning on this SDK/runtime accepts models up to IR v9.
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


class Builder:
    def __init__(self):
        self.nodes = []
        self.initializers = []
        self.counter = 0

    def name(self, prefix):
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def init(self, name, array):
        self.initializers.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def scalar(self, value):
        return self.init(self.name("const"), np.array(value, dtype=np.float32))

    def int64_vec(self, values):
        return self.init(self.name("idx"), np.array(values, dtype=np.int64))

    def op(self, op_type, inputs, outputs=1, **attrs):
        outs = [self.name(op_type.lower()) for _ in range(outputs)]
        self.nodes.append(helper.make_node(op_type, inputs, outs, **attrs))
        return outs[0] if outputs == 1 else outs

    def add(self, a, b):
        return self.op("Add", [a, b])

    def mul(self, a, b):
        return self.op("Mul", [a, b])

    def sub(self, a, b):
        return self.op("Sub", [a, b])

    def div(self, a, b):
        return self.op("Div", [a, b])

    def clip(self, x, lo, hi):
        return self.op("Clip", [x, self.scalar(lo), self.scalar(hi)])

    def slice(self, x, start, end):
        return self.op("Slice", [x, self.int64_vec([start]), self.int64_vec([end]), self.int64_vec([1])])

    def linear(self, x, prefix, weights, bias):
        w_name = self.init(f"{prefix}_W", weights)
        b_name = self.init(f"{prefix}_b", bias)
        return self.add(self.op("MatMul", [x, w_name]), b_name)

    def tanh_approx(self, x):
        n0 = self.scalar(952.52801514)
        n1 = self.scalar(96.39235687)
        n2 = self.scalar(0.60863042)
        d0 = self.scalar(952.72399902)
        d1 = self.scalar(413.36801147)
        d2 = self.scalar(11.88600922)
        x2 = self.mul(x, x)
        num = self.add(self.mul(self.add(self.mul(n2, x2), n1), x2), n0)
        den = self.add(self.mul(self.add(self.mul(d2, x2), d1), x2), d0)
        return self.clip(self.div(self.mul(num, x), den), -1.0, 1.0)

    def sigmoid_approx(self, x):
        return self.add(self.scalar(0.5), self.mul(self.scalar(0.5), self.tanh_approx(self.mul(self.scalar(0.5), x))))

    def gru_cell(self, x, state, prefix, weights):
        hidden = 384
        input_lin = self.linear(x, f"{prefix}_input", *weights[f"{prefix}_input"])
        recur_lin = self.linear(state, f"{prefix}_recurrent", *weights[f"{prefix}_recurrent"])
        z = self.sigmoid_approx(self.add(self.slice(input_lin, 0, hidden), self.slice(recur_lin, 0, hidden)))
        r = self.sigmoid_approx(self.add(self.slice(input_lin, hidden, 2 * hidden), self.slice(recur_lin, hidden, 2 * hidden)))
        h_in = self.slice(input_lin, 2 * hidden, 3 * hidden)
        h_rec = self.slice(recur_lin, 2 * hidden, 3 * hidden)
        h = self.tanh_approx(self.add(h_in, self.mul(h_rec, r)))
        return self.add(self.mul(z, state), self.mul(self.sub(self.scalar(1.0), z), h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-c", default="src/rnnoise_data.c")
    parser.add_argument("--output", default="models/rnnoise_rnn.onnx")
    parser.add_argument("--no-optimize", action="store_true", help="write the raw exported graph without ONNX Runtime optimization")
    args = parser.parse_args()

    text = Path(args.data_c).read_text(encoding="utf-8")
    weights = {}
    for name, (nb_inputs, nb_outputs, sparse, recurrent) in LAYERS.items():
        weights[name] = dense_matrix(text, name, nb_inputs, nb_outputs, sparse, recurrent)

    b = Builder()
    conv1_in = b.op("Concat", ["conv1_state", "features"], axis=1)
    conv1 = b.tanh_approx(b.linear(conv1_in, "conv1", *weights["conv1"]))
    next_conv1 = b.slice(conv1_in, 65, 195)

    conv2_in = b.op("Concat", ["conv2_state", conv1], axis=1)
    conv2 = b.tanh_approx(b.linear(conv2_in, "conv2", *weights["conv2"]))
    next_conv2 = b.slice(conv2_in, 128, 384)

    gru1 = b.gru_cell(conv2, "gru1_state", "gru1", weights)
    gru2 = b.gru_cell(gru1, "gru2_state", "gru2", weights)
    gru3 = b.gru_cell(gru2, "gru3_state", "gru3", weights)
    cat = b.op("Concat", [conv2, gru1, gru2, gru3], axis=1)

    gains = b.sigmoid_approx(b.linear(cat, "dense_out", *weights["dense_out"]))
    vad = b.sigmoid_approx(b.linear(cat, "vad_dense", *weights["vad_dense"]))
    b.nodes.append(helper.make_node("Identity", [gains], ["gains"]))
    b.nodes.append(helper.make_node("Identity", [vad], ["vad"]))
    b.nodes.append(helper.make_node("Identity", [next_conv1], ["next_conv1_state"]))
    b.nodes.append(helper.make_node("Identity", [next_conv2], ["next_conv2_state"]))
    b.nodes.append(helper.make_node("Identity", [gru1], ["next_gru1_state"]))
    b.nodes.append(helper.make_node("Identity", [gru2], ["next_gru2_state"]))
    b.nodes.append(helper.make_node("Identity", [gru3], ["next_gru3_state"]))

    inputs = [
        helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 65]),
        helper.make_tensor_value_info("conv1_state", TensorProto.FLOAT, [1, 130]),
        helper.make_tensor_value_info("conv2_state", TensorProto.FLOAT, [1, 256]),
        helper.make_tensor_value_info("gru1_state", TensorProto.FLOAT, [1, 384]),
        helper.make_tensor_value_info("gru2_state", TensorProto.FLOAT, [1, 384]),
        helper.make_tensor_value_info("gru3_state", TensorProto.FLOAT, [1, 384]),
    ]
    outputs = [
        helper.make_tensor_value_info("gains", TensorProto.FLOAT, [1, 32]),
        helper.make_tensor_value_info("vad", TensorProto.FLOAT, [1, 1]),
        helper.make_tensor_value_info("next_conv1_state", TensorProto.FLOAT, [1, 130]),
        helper.make_tensor_value_info("next_conv2_state", TensorProto.FLOAT, [1, 256]),
        helper.make_tensor_value_info("next_gru1_state", TensorProto.FLOAT, [1, 384]),
        helper.make_tensor_value_info("next_gru2_state", TensorProto.FLOAT, [1, 384]),
        helper.make_tensor_value_info("next_gru3_state", TensorProto.FLOAT, [1, 384]),
    ]
    graph = helper.make_graph(
        b.nodes,
        "rnnoise_rnn_streaming",
        inputs,
        outputs,
        initializer=b.initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    # Windows.AI.MachineLearning on this SDK/runtime accepts models up to IR v9.
    model.ir_version = 9
    onnx.checker.check_model(model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.no_optimize:
        onnx.save(model, output_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            raw_path = Path(tmp.name)
        try:
            onnx.save(model, raw_path)
            optimize_model(raw_path, output_path)
        finally:
            try:
                raw_path.unlink(missing_ok=True)
            except PermissionError:
                warnings.warn(f"could not remove temporary ONNX file: {raw_path}", RuntimeWarning)
    print(args.output)


if __name__ == "__main__":
    main()
