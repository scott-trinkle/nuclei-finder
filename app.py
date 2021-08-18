import base64
import matplotlib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from io import BytesIO
from zipfile import ZipFile
from skimage.io import imread
from scipy.ndimage import binary_erosion

from predict import predict

_lock = RendererAgg.lock


@st.cache(show_spinner=False)
def cache_predict(imgs):
    return predict(imgs)


def plot_image(img, ax, **kwargs):
    ax.imshow(img, interpolation='none', **kwargs)
    ax.set_aspect('equal')
    ax.axis('off')


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None},
          show_spinner=False)
def plot_segmentation(img, pred):
    bounds = pred.astype(int) - binary_erosion(pred).astype(int)
    mask = np.ma.masked_array(bounds, bounds == 0)
    ny, nx = img.shape[:2]
    figw = 5
    figh = figw * ny/nx
    fig, ax = plt.subplots(figsize=(figw,figh))
    plot_image(img, ax)
    plot_image(mask, ax, cmap='Set1')
    fig.tight_layout()
    return fig


class App:
    def __init__(self):
        self.uploaded_files = []
        self.titles = []
        self.N_imgs = None
        self.imgs = None
        self.predictions = None

    def build_sidebar(self):
        intro_text='''
        # Nuclei Finder

        Nuclei Finder is an app for automatic nucleus segmentation in optical
        cell imaging. It was build using a UNET deep learning architecture
        trained from scratch on the [Data Science Bowl
        2018](https://www.kaggle.com/c/data-science-bowl-2018/overview) dataset,
        which includes a diverse set of cell types, magnifications, and imaging
        modalities.

        Made by [Scott Trinkle](https://www.scotttrinkle.com). Code and
        information on the training process is available [on Github]().  '''

        instruct_text='''
        ## Load local images

        To load local images, click "Browse files" or drag and drop image files.
        Images must be <200MB and either in .jpg or .png format.

        ## Load sample images

        To demonstrate the app on sample images, check the "Use sample images"
        box. 

        ## Save results
        
        Enter a custom name for the results folder. Segmented image masks are
        zipped and can be downloaded once the segmentation is complete. 

        ## Run segmentation

        Click "Run segmentation" to calculate the nuclei masks for the currently
        selected images. This launches the results viewer as well as a link 
        to download a zip folder of nuclei masks.
        '''
        st.sidebar.markdown(intro_text)
        instructions = st.sidebar.expander('Usage instructions')
        instructions.markdown(instruct_text)

    def build_controls(self):

        form = st.form(key='controls')
        self.uploaded_files = form.file_uploader('Upload images',
                                                 type=['png', 'jpg'],
                                                 accept_multiple_files=True)
        self.use_sample = form.checkbox('Use sample images',0)
        self.filename = form.text_input('Enter name for zip folder:', value='results.zip')
        form.form_submit_button('Run segmentation')

    def load_imgs(self):
        if not self.use_sample:
            self.imgs = [imread(uploaded_file)
                     for uploaded_file in self.uploaded_files]
            self.titles = [fn.name for fn in self.uploaded_files]
        else:
            fns = [f'sample_imgs/img_{i+1}.png' for i in range(5)]
            self.imgs = [imread(fn) for fn in fns]
            self.titles = [fn.split('/')[1] for fn in fns]
        self.N_imgs = len(self.imgs)
        self.titles_short = [f'{title[:5]}---.png' if len(title) > 9 else title for title in self.titles ]

    def make_predictions(self):
        self.predictions = cache_predict(self.imgs)

    def plot_predictions(self):

        if self.N_imgs == 1:
            with _lock:
                fig = plot_segmentation(self.imgs[0],
                                        self.predictions[0])
                st.write(self.titles_short[0])
                st.pyplot(fig)
        elif self.N_imgs > 1:
            n_cols = 2
            cols = st.columns(n_cols)
            n_rows = int(np.ceil(self.N_imgs/n_cols))
            for row in range(n_rows):
                for col in range(n_cols):
                    ind = row * n_cols + col
                    if ind < self.N_imgs:
                        fig = plot_segmentation(self.imgs[ind],
                                                self.predictions[ind])
                        cols[col].write(self.titles_short[ind])
                        cols[col].pyplot(fig)

    def save_predictions(self):

        with ZipFile('results.zip', 'w') as z:
            for pred, fn in zip(self.predictions, self.titles):
                mask = pred.astype(np.uint8)
                ID, ext = fn.split('.')
                outfn = f'{ID}_mask.{ext}'
                buf = BytesIO()
                plt.imsave(buf, mask, format=ext, cmap='gray')
                z.writestr(outfn, buf.getvalue())

        with open('results.zip', 'rb') as f:
            bytes = f.read()
            b64 = base64.b64encode(bytes).decode()
            href = f"<a href=\"data:file/zip;base64,{b64}\" download={self.filename}>Click to download {self.filename}</a>"
        return href

    def run(self):
        self.build_sidebar()
        
        st.header('Controls')
        self.build_controls()
        results = st.empty()
        
        if (len(self.uploaded_files) > 0) | self.use_sample:
            self.load_imgs()
            results.warning('Running...')
            self.make_predictions()
            href = self.save_predictions()
            results.markdown(href, unsafe_allow_html=True)
            st.header('Results')            
            self.plot_predictions()
        else:
            results.info('Please upload image files or check "Use sample images"')


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
