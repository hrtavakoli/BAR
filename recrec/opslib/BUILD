package(default_visibility = ["//recrec:internal"])

py_library(
    name = "imgprocess",
    srcs = ["imgprocess.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "inputdata",
    srcs = ["inputdata.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "sequenceprocessing",
    srcs = ["sequenceprocessing.py"],
    srcs_version = "PY2AND3",
    deps = [
        "imgprocess"
    ]
)

py_library(
    name = "encode_image",
    srcs = ["encode_image.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "encode_image_test",
    size = "small",
    srcs = ["encode_image_test.py"],
    deps = [
        "encode_image",
    ],
)

