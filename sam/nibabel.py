import os 
import nibabel as nib

class Nibabel:
    def __init__(self, filepath):
        self.proxy = self.load_image(self, filepath)

        # get header
        if self.proxy:
            self.header = self.proxy.header
            if dtype is not None:
                self.dtype = dtype
            else:
                self.dtype = self.header['datatype']
        elif first_dim and second_dim and third_dim and dtype:
            self.header = generate_header(first_dim, second_dim,
                                          third_dim, dtype)
            self.dtype = dtype
        else:
            raise ValueError('Cannot generate a header \
                                (probably missing some argument).')

        self.affine = self.header.get_best_affine()
        self.header_size = self.header.single_vox_offset


    def load_image(self, filepath):

        """Load image into nibabel
        Keyword arguments:
        filepath            : The absolute or relative path
                              of the image
        """

        if not os.path.isfile(filepath):
            logging.warn("File does not exist. "
                         "Will only be able to reconstruct image...")
            return None

        try:
            return nib.load(filepath)
        except Exception as e:
            print("ERROR: Unable to load image into nibabel")
            sys.exit(1)


    def read_data(self, start_pos, end_pos):
        return self.proxy.dataobj[start_pos[0]:end_pos[0],
                                start_pos[1]:end_pos[1],
                                start_pos[2]:end_pos[2]]

    
    def write(to_file, data_array, write_offset, order):
        data = data_array.tobytes(order)
        fd = os.open(to_file, os.O_RDWR | os.O_APPEND)
        os.pwrite(fd, data, write_offset)
        os.close(fd)
        del data_array
        del data