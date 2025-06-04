"""
GPU-optimized main application for the influencer content chatbot creator.
Integrates all components with CUDA acceleration and performance monitoring.
"""

import torch
import argparse
import logging
from models.image_captioner import ImageCaptioner
from models.image_to_video import ImageToVideoGenerator
from models.wan_method import WANVideoPromptGenerator
from config import MODEL_CONFIGS
from utils.gpu import print_gpu_info, optimize_memory, memory_stats
from utils.perf import start_monitoring, stop_monitoring, print_current_status, get_summary_stats
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InfluencerContentCreator:
    def __init__(self):
        """Initialize all components with GPU optimization."""
        logger.info("Initializing GPU-Optimized Influencer Content Creator...")
        
        # Print GPU information
        print_gpu_info()
        
        # Start performance monitoring
        start_monitoring(interval=2.0)
        
        # Initialize image captioner with GPU optimization
        logger.info("Loading image captioning model...")
        self.captioner = ImageCaptioner(MODEL_CONFIGS["image_captioning"])
        
        # Optimize memory after loading first model
        optimize_memory()
        
        # Initialize image-to-video generator with GPU optimization
        logger.info("Loading image-to-video model...")
        self.video_generator = ImageToVideoGenerator(MODEL_CONFIGS["image_to_video"])
        
        # Optimize memory after loading second model
        optimize_memory()
        
        # Initialize WAN method (no GPU needed for this component)
        logger.info("Initializing WAN method...")
        self.wan_generator = WANVideoPromptGenerator()
        
        logger.info("All components loaded successfully!")
        
        # Print current GPU status
        print_current_status()
    
    def process_image(self, image_path, content_type="lifestyle", style="cinematic"):
        """
        Process an image through the complete GPU-optimized pipeline.
        """
        results = {}
        
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Step 1: Generate proper image caption with GPU acceleration
            logger.info("Generating image caption...")
            caption = self.captioner.caption_image(image_path)
            results["caption"] = caption
            logger.info(f"Caption: {caption}")
            
            # Optimize memory between operations
            optimize_memory()
            
            # Step 2: Generate video prompt with GPU acceleration
            logger.info("Generating video prompt...")
            video_prompt = self.video_generator.generate_video_prompt(image_path, style)
            results["video_prompt"] = video_prompt
            logger.info(f"Video Prompt: {video_prompt}")
            
            # Step 3: Generate detailed WAN method prompt
            logger.info("Generating detailed WAN prompt...")
            wan_prompt = self.wan_generator.generate_detailed_prompt(
                subject_description=caption,
                content_type=content_type,
                duration=15,
                style_preference=style
            )
            results["wan_prompt"] = wan_prompt
            logger.info(f"WAN Prompt generated (length: {len(wan_prompt)} chars)")
            
            # Step 4: Generate scene breakdown for longer content
            logger.info("Generating scene breakdown...")
            scenes = self.wan_generator.generate_scene_breakdown(wan_prompt, num_scenes=3)
            results["scene_breakdown"] = scenes
            
            # Final memory optimization
            optimize_memory()
            
            # Print current memory stats
            stats = memory_stats()
            logger.info(f"Current GPU memory usage: {stats}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {"error": str(e)}
    
    def test_functionality(self):
        """Test all functionality with GPU optimization monitoring."""
        logger.info("\n=== Testing GPU-Optimized Functionality ===")
        
        # Create a test image
        test_image = Image.new('RGB', (512, 512), color='blue')
        test_image.save('/tmp/test_image.jpg')
        
        # Test image captioning with performance monitoring
        logger.info("\n1. Testing GPU-Optimized Image Captioning:")
        caption = self.captioner.caption_image('/tmp/test_image.jpg')
        logger.info(f"Generated caption: {caption}")
        logger.info(f"Captioner model info: {self.captioner.get_model_info()}")
        
        # Test image-to-video prompt generation
        logger.info("\n2. Testing GPU-Optimized Image-to-Video:")
        video_prompt = self.video_generator.generate_video_prompt('/tmp/test_image.jpg')
        logger.info(f"Generated video prompt: {video_prompt}")
        logger.info(f"Video generator model info: {self.video_generator.get_model_info()}")
        
        # Test WAN method
        logger.info("\n3. Testing WAN Method:")
        wan_prompt = self.wan_generator.generate_detailed_prompt(
            subject_description="A person in a blue environment",
            content_type="lifestyle",
            duration=15,
            style_preference="cinematic"
        )
        logger.info(f"Generated WAN prompt (first 300 chars): {wan_prompt[:300]}...")
        
        # Print final performance summary
        logger.info("\n=== Performance Summary ===")
        print_current_status()
        
        logger.info("\n=== All GPU-optimized tests completed successfully! ===")
    
    def get_performance_report(self):
        """Get comprehensive performance report."""
        # Stop monitoring and get summary
        monitoring_data = stop_monitoring()
        summary = get_summary_stats()
        
        report = {
            "monitoring_summary": summary,
            "current_gpu_stats": memory_stats(),
            "model_info": {
                "captioner": self.captioner.get_model_info(),
                "video_generator": self.video_generator.get_model_info()
            }
        }
        
        return report
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            # Stop monitoring
            stop_monitoring()
            # Final memory cleanup
            optimize_memory()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description='GPU-Optimized Influencer Content Chatbot Creator')
    parser.add_argument('--test', action='store_true', help='Run functionality tests')
    parser.add_argument('--image', type=str, help='Path to image to process')
    parser.add_argument('--content-type', type=str, default='lifestyle', 
                       choices=['lifestyle', 'fashion', 'business', 'tutorial'],
                       help='Type of content to create')
    parser.add_argument('--style', type=str, default='cinematic',
                       choices=['cinematic', 'social_media', 'artistic'],
                       help='Style preference for video generation')
    parser.add_argument('--gpu-info', action='store_true', help='Show GPU information and exit')
    parser.add_argument('--performance-report', action='store_true', help='Generate performance report')
    
    args = parser.parse_args()
    
    if args.gpu_info:
        # Just show GPU info and exit
        print_gpu_info()
        return
    
    # Initialize the creator
    creator = InfluencerContentCreator()
    
    try:
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
            print("Use --test to run tests, --image <path> to process an image, or --gpu-info for GPU details")
        
        if args.performance_report:
            # Generate performance report
            report = creator.get_performance_report()
            print("\n=== Performance Report ===")
            print(f"Monitoring Summary: {report['monitoring_summary']}")
            print(f"Current GPU Stats: {report['current_gpu_stats']}")
            print(f"Model Info: {report['model_info']}")
    
    finally:
        # Ensure cleanup
        del creator

if __name__ == "__main__":
    main()
