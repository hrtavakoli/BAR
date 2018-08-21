package(default_visibility = [":internal"])


package_group(
    name = "internal",
    packages = [
        "//recrec/...",
    ],
)


py_library(
    name = "imgRecRecModel",
    srcs = ["imgRecRecModel.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//recrec/opslib:encode_image",
        "//recrec/opslib:imgprocess",
        "//recrec/opslib:inputdata",
    ],
)

py_library(
    name = "imgRecModel",
    srcs = ["imgRecModel.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//recrec/opslib:encode_image",
        "//recrec/opslib:imgprocess",
        "//recrec/opslib:inputdata",
    ],
)


py_library(
    name = "configuration",
    srcs = ["configuration.py"],
    srcs_version = "PY2AND3",
)

py_binary(
    name = "train_image_recognition",
    srcs = ["train_image_recognition.py"],
    srcs_version = "PY2AND3",
    deps = [
        "imgRecRecModel",
        "configuration",
    ],
)

py_binary(
    name = "train_cnn_recognition",
    srcs = ["train_cnn_recognition.py"],
    srcs_version = "PY2AND3",
    deps = [
        "imgRecModel",
        "configuration",
    ],
)


py_binary(
    name = "predict",
    srcs = ["predict.py"],
    srcs_version = "PY2AND3",
    deps = [
        "imgRecRecModel",
        "configuration",
    ],
)

py_binary(
    name = "predict_recurrent_cifar",
    srcs = ["predict_recurrent_cifar.py"],
    srcs_version = "PY2AND3",
    deps = [
        "imgRecRecModel",
        "configuration",
    ],
)


py_binary(
    name = "predict_cnn",
    srcs = ["predict_cnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        "imgRecModel",
        "configuration",
    ],
)

py_binary(
    name = "predict_cnn_cifar",
    srcs = ["predict_cnn_cifar.py"],
    srcs_version = "PY2AND3",
    deps = [
        "imgRecModel",
        "configuration",
    ],
)
