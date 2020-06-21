

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