""
Configuration file for the influencer content chatbot creator.
Contains model paths and settings.
"""

# Fixed model repository paths - using verified working models
MODEL_CONFIGS = {
    "text_generation": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "torch_dtype": "float16",
        "load_in_4bit": True,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    },
    "image_captioning": {
        "model_id": "Salesforce/blip2-opt-2.7b",
        "torch_dtype": "float16",
        "max_new_tokens": 100,
        "temperature": 0.3,
        "repetition_penalty": 1.2,
        "do_sample": True
    },
    "image_to_video": {
        "model_id": "ali-vilab/text-to-video-ms-1.7b",
        "torch_dtype": "float16",
        "num_inference_steps": 25,
        "guidance_scale": 9.0
    }
}

# WAN (Wide Area Network) method configuration for detailed video prompts
WAN_PROMPT_TEMPLATE = """
Create a highly detailed video prompt with the following elements:

SCENE DESCRIPTION:
- Setting: {setting}
- Time of day: {time_of_day}
- Weather/atmosphere: {atmosphere}

CAMERA WORK:
- Shot type: {shot_type}
- Camera movement: {camera_movement}
- Angle: {camera_angle}

LIGHTING:
- Primary lighting: {lighting_type}
- Mood: {lighting_mood}
- Color temperature: {color_temp}

SUBJECT/CHARACTER:
- Appearance: {subject_description}
- Action: {action}
- Emotion: {emotion}
- Clothing/style: {clothing}

TECHNICAL SPECS:
- Duration: {duration} seconds
- Frame rate: {fps} fps
- Resolution: {resolution}
- Style: {visual_style}

AUDIO ELEMENTS:
- Background music: {music_style}
- Sound effects: {sound_effects}
- Ambient sounds: {ambient_sounds}

NARRATIVE ELEMENTS:
- Story beat: {story_beat}
- Pacing: {pacing}
- Transition: {transition_type}
""