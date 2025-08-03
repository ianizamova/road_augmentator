class AugmentationPipeline:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.segmenter = RoadSegmenter(self.config['segmenter'])
        self.generator = ObjectGenerator(self.config['generator'])
    
    def run(self):
        for bg_image in self.load_backgrounds():
            road_mask = self.segmenter.segment(bg_image)
            obj = self.generator.generate("a horse")
            result = self.blender.blend(bg_image, obj, road_mask)
            self.save_result(result)