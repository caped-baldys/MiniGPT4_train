Customizing MiniGPT4-video for your own Video-text dataset
1.Add your own video dataloader
Construct your own dataloader here minigpt4/datasets/datasets/video_datasets.py based on the existing dataloaders.
Copy Video_loader_template class and edit it according to you data nature. 
(line 980)
class Video_loader_template(BaseDataset, __DisplMixin):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,subtitles_path,model_name='llama2',add_subtitles=True):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.model_name=model_name
        if self.model_name =='mistral':
            self.length = 90
####> line(1011) reads video_id column
def __getitem__(self, index):
        ann = self.annotation[index]
        video_id = ann["video_id"] # video_id
        answer=ann["a"] # answer (ground truth)
        instruction=ann["q"] # question (instruction)
        images=[]
        img_placeholder = ""
        has_subtitles = self.videos_has_subtitles.get(video_id, False)
        if self.add_subtitles and has_subtitles:
            subtitle_path = os.path.join(self.subtitle_folder, f'{video_id}.vtt')
            # Load the VTT subtitle file
            vtt_file = webvtt.read(subtitle_path)
                
        video_path = os.path.join(self.vis_root,'videos',f'{video_id}.{self.videos_extension[video_id]}')
	##Sample video path '/path/to/your/vis_root/video_id.mp4'
        clip = VideoFileClip(video_path)
...............
########### 
line(55)
def generate_subtitles(video_path,existed_subtitles):
    video_id=video_path.split('/')[-1].split('.')[0]
    audio_path = f"workspace/misssing_eval_subtitles/mp3/{video_id}"+'.mp3'
    if existed_subtitles.get(video_id,False):
        print("subtitle already generated")
        return f"workspace/misssing_eval_subtitles/{video_id}"+'.vtt'
    try:
        extract_audio(video_path,audio_path)
        print("successfully extracted")
        os.system(f"whisper {audio_path}  --language English --model large --output_format vtt --output_dir workspace/misssing_eval_subtitles")
        # remove the audio file
        os.system(f"rm {audio_path}")
        print("subtitle successfully generated")  
        return f"workspace/misssing_eval_subtitles/{video_id}"+'.vtt'
.................

2.Create config file for your dataloader
Here minigpt4/configs/datasets/dataset_name/default.yaml creates your yaml file that includes paths to your dataset.
Copy the template file minigpt4/configs/datasets/template/default.yaml and edit the paths to your dataset.
default.yml
>
datasets:
  dataset_name: # same as the name of the train_config yaml file
    # data_dir: ${env.data_dir}/datasets
    data_type: images # let it be images for now even if it is videos

    build_info: # this is the information needed to build the dataset
      # Be careful not to append minus sign (-) before split to avoid itemizing
      ann_paths: [path/to/annotations_json] # list of paths to annotation files
      vis_root: path/to/videos_folder
      subtitles_path:	2 path/to/subtitles_folder
      model_name: 'llama2' # Language Model Name (available: llama2, mistral)
.....................
3.Register your dataloader
In the minigpt4/datasets/builders/image_text_pair_builder.py file Import your data loader class from the minigpt4/datasets/datasets/video_datasets.py file
Copy and edit the VideoTemplateBuilder class.
put the train_dataset_cls = YourVideoLoaderClass that you imported from minigpt4/datasets/datasets/video_datasets.py file.

......................
4.Edit training config file
Add your dataset to the datasets in the yml file as shown below:

datasets:
  dataset_name: # change this to your dataset name
    batch_size: 4  # change this to your desired batch size
    vis_processor:
      train:
        name: "blip2_image_train"
        image_size: 224
    text_processor:
      train:
        name: "blip_caption"
    sample_ratio: 200 # if you including joint training with other datasets, you can set the sample ratio here
	resume_ckpt_path : "path_to_pretrained_model_for_fine_tuning_it"
.....................................................




🔥 Training
To customize MiniGPT4-Video for your own Video-text dataset
You can find the steps to customize MiniGPT4-Video for your own video-text dataset in Custom_training.md

Training datasets
After downloading the datasets below, you should go to the datasets configuration folder here minigpt4/configs/datasets set the paths for each dataset there.
Image text training
You can find the steps to download the datasets in MiniGPT4

LAION
Conceptual Captions
SBU
Video text training:

CMD
Webvid
Video Instructional Dataset 100K
You can find the datasets annotation files for video_text datasets here download

Model training:
You can edit the number of gpus in the each script.sh below

Stage 1 (image text pretraining)
You can directly download the pretrained MiniGPT4 checkpoint aligned with Llama2.

Or train by yourself:

# pretrain
# Llama2
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/224_minigpt4_llama2_image.yaml
# Mistral
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/224_minigpt4_mistral_image.yaml

# align
# To launch the second stage alignment, first specify the path to the checkpoint file trained in pretrain stage.
# Llama2
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/224_minigpt4_llama2_image_align.yaml
# Mistral
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/224_minigpt4_mistral_image_align.yaml
You can download our trained weights for this stage from here Llama2 Mistral

Stage 2 (video captioning pretraining)
For Llama2
set the cfg-path in the script to train_configs/224_v2_llama2_video_stage_2.yaml
set the model name here minigpt4/configs/datasets/cmd_video/default.yaml and minigpt4/configs/datasets/webvid/default.yaml to llama2
For Mistral
set the cfg-path in the script to train_configs/224_v2_mistral_video_stage_2.yaml
set the model name here minigpt4/configs/datasets/cmd_video/default.yaml and minigpt4/configs/datasets/webvid/default.yaml to mistral

bash jobs_video/train/stage_2.sh
You can download our trained weights for this stage from here Llama2 Mistral



Stage 3 (video Instruction finetuning)
For Llama2
set the cfg-path in the script to train_configs/224_v2_llama2_video_stage_3.yaml
set the model name here minigpt4/configs/datasets/video_chatgpt/default.yaml to llama2

For Mistral
set the cfg-path in the script to train_configs/224_v2_mistral_video_stage_3.yaml
set the model name here minigpt4/configs/datasets/video_chatgpt/default.yaml to mistral

bash jobs_video/train/stage_3.sh
You can download our trained weights for this stage from here Llama2 Mistral


Instruction Pool

Describe this video.',
            'Provide a concise depiction of this video.',
            'Present a description of this video.',
            'Summarize this video.',
            'Generate video caption:',
            'Generate video description:',
            'Write a description for the video.',
            'Provide a description of what is presented in the video.',
            'Describe the content of the video.',
            'Can you explain what you see in the video?',
            'Could you describe what you perceive in the video?',
            'Please provide a depiction of the video.',
            'Illustrate what is happening in the video.',