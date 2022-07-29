import imageio
from .context import Context


class Recorder(Context):

    writer = None

    recording = False

    def stimulate(self, *args, **kwargs):
        super().stimulate(*args, **kwargs)
        if self.writer and 'to_set' not in kwargs.keys():
            self.capture_frame()

    def add_definition(self, *args, **kwargs):
        res = super().add_definition(*args, **kwargs)
        self.capture_frame()
        return res

    def add_text(self, *args, **kwargs):
        res = super().add_text(*args, **kwargs)
        self.capture_frame()
        return res

    def capture_frame(self):
        if self.recording:
            self.render('./frame.png', self.title, self.consider_stimulus, self.arrow_size)
            image = imageio.imread('./frame.png')
            self.writer.append_data(image)

    def start_recording(self, filename, title, consider_stimulus, fps = 12, arrow_size = 3):
        self.title = title
        self.recording = True
        self.consider_stimulus = consider_stimulus
        self.arrow_size = arrow_size
        self.writer = imageio.get_writer(filename, mode='I', fps=fps)
        self.capture_frame()

    def stop_recording(self):
        self.writer.close()
        self.recording = False
        self.write = None
