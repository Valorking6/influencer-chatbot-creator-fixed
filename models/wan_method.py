"""
WAN (Wide Area Network) method for generating highly detailed video prompts.
This method produces comprehensive, scene-by-scene video descriptions.
"""

import random
from typing import Dict, List, Any
from config import WAN_PROMPT_TEMPLATE

class WANVideoPromptGenerator:
    def __init__(self):
        self.prompt_template = WAN_PROMPT_TEMPLATE
        self.style_database = self._initialize_style_database()
    
    def _initialize_style_database(self):
        """Initialize comprehensive style database for detailed prompts."""
        return {
            "settings": [
                "luxurious penthouse apartment", "bustling city street", "serene beach at sunset",
                "modern office space", "cozy coffee shop", "elegant restaurant", "rooftop terrace",
                "art gallery", "fashion studio", "outdoor park", "mountain landscape", "urban loft"
            ],
            "time_of_day": [
                "golden hour", "blue hour", "midday sun", "early morning", "late evening",
                "dawn", "dusk", "midnight", "afternoon", "sunrise", "sunset"
            ],
            "atmosphere": [
                "warm and inviting", "cool and mysterious", "bright and energetic", "moody and dramatic",
                "soft and romantic", "crisp and clean", "hazy and dreamy", "vibrant and colorful"
            ],
            "shot_types": [
                "close-up portrait", "medium shot", "wide establishing shot", "extreme close-up",
                "full body shot", "over-the-shoulder", "bird's eye view", "low angle", "high angle"
            ],
            "camera_movements": [
                "smooth dolly push-in", "elegant crane movement", "steady tracking shot",
                "subtle handheld movement", "slow zoom", "circular orbit", "reveal shot",
                "pull-back reveal", "push-in focus", "lateral tracking"
            ],
            "camera_angles": [
                "eye level", "slightly low angle", "high angle", "dutch angle", "overhead",
                "ground level", "three-quarter angle", "profile angle", "frontal angle"
            ],
            "lighting_types": [
                "soft natural window light", "dramatic key lighting", "rim lighting",
                "three-point lighting", "ambient lighting", "practical lighting",
                "colored gel lighting", "hard directional light", "diffused lighting"
            ],
            "lighting_moods": [
                "warm and flattering", "cool and modern", "dramatic and contrasty",
                "soft and ethereal", "vibrant and energetic", "moody and atmospheric",
                "bright and airy", "intimate and cozy"
            ],
            "color_temps": [
                "warm 3200K", "daylight 5600K", "cool 7000K", "mixed temperature",
                "golden 2800K", "neutral 4000K", "blue hour 8000K"
            ],
            "actions": [
                "confidently walking", "gracefully dancing", "thoughtfully speaking",
                "elegantly posing", "naturally laughing", "professionally presenting",
                "casually interacting", "dynamically moving", "expressively gesturing"
            ],
            "emotions": [
                "confident and empowered", "joyful and radiant", "serene and peaceful",
                "passionate and intense", "playful and energetic", "sophisticated and elegant",
                "warm and approachable", "mysterious and alluring", "determined and focused"
            ],
            "clothing_styles": [
                "elegant evening wear", "casual chic outfit", "professional business attire",
                "trendy streetwear", "bohemian flowing dress", "minimalist modern look",
                "vintage inspired ensemble", "athletic wear", "artistic avant-garde piece"
            ],
            "visual_styles": [
                "cinematic film look", "high-fashion editorial", "documentary realism",
                "artistic and stylized", "commercial polished", "indie film aesthetic",
                "music video dynamic", "social media optimized", "luxury brand quality"
            ],
            "music_styles": [
                "upbeat electronic", "ambient atmospheric", "classical orchestral",
                "modern pop", "jazz instrumental", "acoustic indie", "cinematic score",
                "lo-fi chill", "energetic dance", "emotional piano"
            ],
            "pacing": [
                "slow and contemplative", "medium steady rhythm", "fast and dynamic",
                "varied pacing", "building intensity", "relaxed flow", "rhythmic cuts"
            ]
        }
    
    def generate_detailed_prompt(self, 
                               subject_description: str,
                               content_type: str = "lifestyle",
                               duration: int = 15,
                               style_preference: str = "cinematic") -> str:
        """
        Generate a highly detailed video prompt using the WAN method.
        
        Args:
            subject_description: Description of the main subject
            content_type: Type of content (lifestyle, fashion, business, etc.)
            duration: Video duration in seconds
            style_preference: Overall style preference
            
        Returns:
            str: Comprehensive, detailed video prompt
        """
        
        # Select appropriate elements based on content type and style
        elements = self._select_elements_for_content(content_type, style_preference)
        
        # Generate technical specifications
        tech_specs = self._generate_tech_specs(duration, style_preference)
        
        # Create the detailed prompt
        detailed_prompt = self.prompt_template.format(
            setting=elements["setting"],
            time_of_day=elements["time_of_day"],
            atmosphere=elements["atmosphere"],
            shot_type=elements["shot_type"],
            camera_movement=elements["camera_movement"],
            camera_angle=elements["camera_angle"],
            lighting_type=elements["lighting_type"],
            lighting_mood=elements["lighting_mood"],
            color_temp=elements["color_temp"],
            subject_description=subject_description,
            action=elements["action"],
            emotion=elements["emotion"],
            clothing=elements["clothing"],
            duration=tech_specs["duration"],
            fps=tech_specs["fps"],
            resolution=tech_specs["resolution"],
            visual_style=elements["visual_style"],
            music_style=elements["music_style"],
            sound_effects=tech_specs["sound_effects"],
            ambient_sounds=tech_specs["ambient_sounds"],
            story_beat=elements["story_beat"],
            pacing=elements["pacing"],
            transition_type=elements["transition_type"]
        )
        
        return detailed_prompt
    
    def _select_elements_for_content(self, content_type: str, style_preference: str) -> Dict[str, str]:
        """Select appropriate style elements based on content type and preference."""
        
        # Content-specific element selection
        if content_type == "fashion":
            settings = ["fashion studio", "urban loft", "rooftop terrace", "art gallery"]
            lighting_types = ["dramatic key lighting", "soft natural window light", "rim lighting"]
            visual_styles = ["high-fashion editorial", "luxury brand quality", "artistic and stylized"]
        elif content_type == "business":
            settings = ["modern office space", "luxurious penthouse apartment", "elegant restaurant"]
            lighting_types = ["three-point lighting", "soft natural window light", "professional lighting"]
            visual_styles = ["commercial polished", "cinematic film look", "professional presentation"]
        elif content_type == "lifestyle":
            settings = ["cozy coffee shop", "serene beach at sunset", "outdoor park", "bustling city street"]
            lighting_types = ["soft natural window light", "ambient lighting", "golden hour lighting"]
            visual_styles = ["documentary realism", "social media optimized", "indie film aesthetic"]
        else:
            # Default to full range
            settings = self.style_database["settings"]
            lighting_types = self.style_database["lighting_types"]
            visual_styles = self.style_database["visual_styles"]
        
        return {
            "setting": random.choice(settings),
            "time_of_day": random.choice(self.style_database["time_of_day"]),
            "atmosphere": random.choice(self.style_database["atmosphere"]),
            "shot_type": random.choice(self.style_database["shot_types"]),
            "camera_movement": random.choice(self.style_database["camera_movements"]),
            "camera_angle": random.choice(self.style_database["camera_angles"]),
            "lighting_type": random.choice(lighting_types),
            "lighting_mood": random.choice(self.style_database["lighting_moods"]),
            "color_temp": random.choice(self.style_database["color_temps"]),
            "action": random.choice(self.style_database["actions"]),
            "emotion": random.choice(self.style_database["emotions"]),
            "clothing": random.choice(self.style_database["clothing_styles"]),
            "visual_style": random.choice(visual_styles),
            "music_style": random.choice(self.style_database["music_styles"]),
            "pacing": random.choice(self.style_database["pacing"]),
            "story_beat": self._generate_story_beat(content_type),
            "transition_type": self._generate_transition_type(style_preference)
        }
    
    def _generate_tech_specs(self, duration: int, style_preference: str) -> Dict[str, Any]:
        """Generate technical specifications for the video."""
        
        # Determine specs based on style and duration
        if style_preference == "cinematic":
            fps = "24"
            resolution = "4K (3840x2160)"
            sound_effects = "subtle ambient sounds, professional foley"
            ambient_sounds = "atmospheric background, environmental audio"
        elif style_preference == "social_media":
            fps = "30"
            resolution = "1080p (1920x1080) vertical"
            sound_effects = "trendy sound effects, modern audio elements"
            ambient_sounds = "upbeat background ambiance"
        else:
            fps = "30"
            resolution = "4K (3840x2160)"
            sound_effects = "natural sound effects, realistic audio"
            ambient_sounds = "environmental sounds, natural ambiance"
        
        return {
            "duration": duration,
            "fps": fps,
            "resolution": resolution,
            "sound_effects": sound_effects,
            "ambient_sounds": ambient_sounds
        }
    
    def _generate_story_beat(self, content_type: str) -> str:
        """Generate appropriate story beat based on content type."""
        story_beats = {
            "fashion": "Showcase the outfit/style transformation",
            "business": "Demonstrate expertise and professionalism",
            "lifestyle": "Capture authentic moment of daily life",
            "tutorial": "Educational demonstration or explanation",
            "promotional": "Product or service highlight",
            "personal": "Intimate personal story or experience"
        }
        return story_beats.get(content_type, "Engaging narrative moment")
    
    def _generate_transition_type(self, style_preference: str) -> str:
        """Generate appropriate transition type based on style."""
        transitions = {
            "cinematic": "Smooth cross-dissolve with motion blur",
            "social_media": "Quick cut with trendy transition effect",
            "artistic": "Creative wipe or morphing transition",
            "documentary": "Natural cut or fade transition",
            "commercial": "Professional slide or push transition"
        }
        return transitions.get(style_preference, "Smooth fade transition")
    
    def generate_scene_breakdown(self, detailed_prompt: str, num_scenes: int = 3) -> List[str]:
        """
        Break down the detailed prompt into multiple scenes for longer videos.
        
        Args:
            detailed_prompt: The main detailed prompt
            num_scenes: Number of scenes to create
            
        Returns:
            List[str]: List of scene-specific prompts
        """
        scenes = []
        
        for i in range(num_scenes):
            # Modify elements for each scene
            scene_elements = self._select_elements_for_content("lifestyle", "cinematic")
            
            scene_prompt = f"Scene {i+1}: {detailed_prompt}"
            # Add scene-specific modifications
            if i == 0:
                scene_prompt += "\n\nOPENING: Establishing shot with gradual reveal"
            elif i == num_scenes - 1:
                scene_prompt += "\n\nCLOSING: Final impactful moment with elegant conclusion"
            else:
                scene_prompt += f"\n\nMIDDLE SCENE: Development and progression of narrative"
            
            scenes.append(scene_prompt)
        
        return scenes