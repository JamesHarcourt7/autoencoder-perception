import struct
import numpy
import operator
from six.moves import reduce

# Source: idx2numpy/converters.py at master · ivanyu/idx2numpy · GitHub

# Keys are IDX data type codes.
# Values: (ndarray data type name, name for struct.unpack, size in bytes).
_DATA_TYPES_IDX = {
    0x08: ('ubyte', 'B', 1),
    0x09: ('byte', 'b', 1),
    0x0B: ('>i2', 'h', 2),
    0x0C: ('>i4', 'i', 4),
    0x0D: ('>f4', 'f', 4),
    0x0E: ('>f8', 'd', 8)
}

# Keys are ndarray data type name.
# Values: (IDX data type code, name for struct.pack, size in bytes).
_DATA_TYPES_NUMPY = {
    'uint8': (0x08, 'B'),
    'int8': (0x09, 'b'),
    'int16': (0x0B, 'h'),
    'int32': (0x0C, 'i'),
    'float32': (0x0D, 'f'),
    'float64': (0x0E, 'd'),
}


def load_data():
    # Parse IDX files into numpy arrays
    train_x = convertidx2numpy('mnist/train-images.idx3-ubyte')
    train_y = convertidx2numpy('mnist/train-labels.idx1-ubyte')
    test_x = convertidx2numpy('mnist/t10k-images.idx3-ubyte')
    test_y = convertidx2numpy('mnist/t10k-labels.idx1-ubyte')
    
    return (train_x, train_y), (test_x, test_y)

def convertidx2numpy(f):
    with open(f, 'rb') as inp:
        # Read the "magic number" - 4 bytes.
        try:
            mn = struct.unpack('>BBBB', inp.read(4))
        except struct.error:
            raise FormatError(struct.error)

        # First two bytes are always zero, check it.
        if mn[0] != 0 or mn[1] != 0:
            msg = ("Incorrect first two bytes of the magic number: " +
                "0x{0:02X} 0x{1:02X}".format(mn[0], mn[1]))
            raise FormatError(msg)

        # 3rd byte is the data type code.
        dtype_code = mn[2]
        if dtype_code not in _DATA_TYPES_IDX:
            msg = "Incorrect data type code: 0x{0:02X}".format(dtype_code)
            raise FormatError(msg)

        # 4th byte is the number of dimensions.
        dims = int(mn[3])

        # See possible data types description.
        dtype, dtype_s, el_size = _DATA_TYPES_IDX[dtype_code]

        # 4-byte integer for length of each dimension.
        try:
            dims_sizes = struct.unpack('>' + 'I' * dims, inp.read(4 * dims))
        except struct.error as e:
            raise FormatError('Dims sizes: {0}'.format(e))

        # Full length of data.
        full_length = reduce(operator.mul, dims_sizes, 1)

        # Create a numpy array from the data
        try:
            result_array = numpy.frombuffer(
                inp.read(full_length * el_size),
                dtype=numpy.dtype(dtype)
            ).reshape(dims_sizes)
        except ValueError as e:
            raise FormatError('Error creating numpy array: {0}'.format(e))

        # Check for superfluous data.
        if len(inp.read(1)) > 0:
            raise FormatError('Superfluous data detected.')

        return result_array
    

if __name__ == "__main__":
    (train_x, train_y), (test_x, test_y) = load_data()
    print(len(train_x))
    print(train_x[0])
