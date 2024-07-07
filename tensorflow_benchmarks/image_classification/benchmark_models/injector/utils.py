import struct


def float32_to_int(value):
    bytes_value = struct.pack("f", value)
    int_value = struct.unpack("I", bytes_value)[0]
    return int_value


def int_to_float32(int_value):
    bytes_value = struct.pack("I", int_value)
    return struct.unpack("f", bytes_value)[0]
