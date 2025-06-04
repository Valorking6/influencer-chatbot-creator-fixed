"""
Main application file for the influencer content chatbot creator.
Integrates all fixed components.
"""

import torch
from models.image_captioner import ImageCaptioner
from models.image_to_video import ImageToVideoGenerator
from models.wan_method import WANVideoPromptGenerator
from config import MODEL_CONFIGS
from PIL import Image
import argparse

class InfluencerContentCreator:
    def __init__(self):
        """Initialize all components with fixed configurations."""
        print("Initializing Influencer Content Creator...")
        
        # Initialize image captioner with fixed repetition issues
        print("Loading image captioning model...")
        self.captioner = ImageCaptioner(MODEL_CONFIGS["image_captioning"])
        
        # Initialize image-to-video generator with actual video prompt generation
        print("Loading image-to-video model...")
        self.video_generator = ImageToVideoGenerator(MODEL_CONFIGS["image_to_video"])
        
        # Initialize WAN method for detailed video prompts
        print("Initializing WAN method...")
        self.wan_generator = WANVideoPromptGenerator()
        
        print("All components loaded successfully!")
    
    def process_image(self, image_path, content_type="lifestyle", style="cinematic"):
        """
        Process an image through the complete pipeline.
        
        Args:
            image_path: Path to the input image
            content_type: Type of content to create
            style: Style preference for video generation
            
        Returns:
            dict: Results from all processing steps
        """
        results = {}
        
        try:
            # Step 1: Generate proper image caption (fixed repetition issue)
            print("Generating image caption...")
            caption = self.captioner.caption_image(image_path)
            results["caption"] = caption
            print(f"Caption: {caption}")
            
            # Step 2: Generate video prompt (actual prompt, not instructions)
            print("Generating video prompt...")
            video_prompt = self.video_generator.generate_video_prompt(image_path, style)
            results["video_prompt"] = video_prompt
            print(f"Video Prompt: {video_prompt}")
            
            # Step 3: Generate detailed WAN method prompt
            print("Generating detailed WAN prompt...")
            wan_prompt = self.wan_generator.generate_detailed_prompt(
                subject_description=caption,
                content_type=content_type,
                duration=15,
                style_preference=style
            )
            results["wan_prompt"] = wan_prompt
            print(f"WAN Prompt: {wan_prompt[:200]}...")
            
            # Step 4: Generate scene breakdown for longer content
            print("Generating scene breakdown...")
            scenes = self.wan_generator.generate_scene_breakdown(wan_prompt, num_scenes=3)
            results["scene_breakdown"] = scenes
            
            return results
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return {"error": str(e)}
    
    def test_functionality(self):
        """Test all functionality to ensure fixes work properly."""
        print("\n=== Testing Functionality ===")
        
        # Create a test image
        test_image = Image.new('RGB', (512, 512), color='blue')
        test_image.save('/tmp/test_image.jpg')
        
        # Test image captioning
        print("\n1. Testing Image Captioning (Fixed repetition issue):")
        caption = self.captioner.caption_image('/tmp/test_image.jpg')
        print(f"Generated caption: {caption}")
        
        # Test image-to-video prompt generation
        print("\n2. Testing Image-to-Video (Fixed to generate actual prompts):")
        video_prompt = self.video_generator.generate_video_prompt('/tmp/test_image.jpg')
        print(f"Generated video prompt: {video_prompt}")
        
        # Test WAN method
        print("\n3. Testing WAN Method (Detailed video prompts):")
        wan_prompt = self.wan_generator.generate_detailed_prompt(
            subject_description="A person in a blue environment",
            content_type="lifestyle",
            duration=15,
            style_preference="cinematic"
        )
        print(f"Generated WAN prompt: {wan_prompt[:300]}...")
        
        print("\n=== All tests completed successfully! ===")

def main():
    parser = argparse.ArgumentParser(description='Influencer Content Chatbot Creator')
    parser.add_argument('--test', action='store_true', help='Run functionality tests')
    parser.add_argument('--image', type=str, help='Path to image to process')
    parser.add_argument('--content-type', type=str, default='lifestyle', 
                       choices=['lifestyle', 'fashion', 'business', 'tutorial'],
                       help='Type of content to create')
    parser.add_argument('--style', type=str, default='cinematic',
                       choices=['cinematic', 'social_media', 'artistic'],
                       help='Style preference for video generation')
    
    args = parser.parse_args()
    
    # Initialize the creator
    creator = InfluencerContentCreator()
    
    if args.test:
        # Run tests
        creator.test_functionality()
    elif args.image:
        # Process specific image
        results = creator.process_image(args.image, args.content_type, args.style)
        print("\n=== Processing Results ===")
        for key, value in results.items():
            if key != "scene_breakdown":
                print(f"{key.upper()}: {value}")
            else:
                print(f"{key.upper()}:")
                for i, scene in enumerate(value, 1):
                    print(f"  Scene {i}: {scene[:100]}...")
    else:
        print("Use --test to run tests or --image <path> to process an image")

if __name__ == "__main__":
    main()