import imageio
from .context import Context


class Recorder(Context):

    writer = None

    recording = False

    def stimulate(self, *args, **kwargs):
        super().stimulate(*args, **kwargs)
        if self.writer and 'to_set' not in kwargs.keys():
            self.capture_frame()

    def decrease_stimulus(self, *args, **kwargs):
        super().decrease_stimulus(*args, **kwargs)
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
            self.render('./frame.png', self.title, self.consider_stimulus,
                        self.arrow_size, skip_empty_nodes=self.skip_empty_nodes, force_text_rendering=self.force_text_rendering)
            image = imageio.imread('./frame.png')
            self.writer.append_data(image)

    def start_recording(self, filename, title, consider_stimulus, skip_empty_nodes=False, fps=12, arrow_size=3, force_text_rendering=False):
        self.title = title
        self.recording = True
        self.force_text_rendering = force_text_rendering
        self.skip_empty_nodes = skip_empty_nodes
        self.consider_stimulus = consider_stimulus
        self.arrow_size = arrow_size
        self.writer = imageio.get_writer(filename, mode='I', fps=fps)
        self.capture_frame()

    def stop_recording(self):
        self.writer.close()
        self.recording = False
        self.write = None
