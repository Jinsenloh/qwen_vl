from ollama import chat
from PIL import Image
import base64
import json
import os

class QwenVLModel:
    def __init__(self, model_name="qwen3-vl:8b"):
        """
        Initialize Qwen VL model with Ollama

        Args:
            model_name (str): Name of the Qwen VL model in Ollama
        """
        self.model_name = model_name

    def encode_image_to_base64(self, image_path):
        """
        Convert image to base64 string

        Args:
            image_path (str): Path to the image file

        Returns:
            str: Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def chat_with_image(self, prompt, image_path=None, image_base64=None):
        """
        Chat with the Qwen VL model using text and optional image

        Args:
            prompt (str): Text prompt for the model
            image_path (str, optional): Path to image file
            image_base64 (str, optional): Base64 encoded image string

        Returns:
            str: Model's response
        """
        messages = [
            {
                'role': 'user',
                'content': prompt
            }
        ]

        # Add image if provided
        if image_path:
            image_base64 = self.encode_image_to_base64(image_path)

        if image_base64:
            messages[0]['images'] = [image_base64]

        try:
            response = chat(
                model=self.model_name,
                messages=messages
            )
            return response['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

    def describe_image(self, image_path):
        """
        Describe what's in an image

        Args:
            image_path (str): Path to the image file

        Returns:
            str: Description of the image
        """
        prompt = "Describe what you see in this image in detail."
        return self.chat_with_image(prompt, image_path=image_path)

    def answer_about_image(self, question, image_path):
        """
        Answer a specific question about an image

        Args:
            question (str): Question about the image
            image_path (str): Path to the image file

        Returns:
            str: Answer to the question
        """
        return self.chat_with_image(question, image_path=image_path)

    def create_activity_classification_prompt(self):
        """
        Create a few-shot prompt for activity classification

        Returns:
            str: Few-shot prompt with examples and instructions
        """
        prompt = """You are an expert human activity classifier. Your task is to classify images into one of three activities: sit, walk, or stand.

CLASSIFICATION GUIDELINES:
- "sit": Person is seated (on chair, bench, ground, etc.)
- "walk": Person is walking, jogging, running, or in motion
- "stand": Person is standing upright, stationary

IMPORTANT INSTRUCTIONS:
1. Analyze the image carefully for human poses and activities
2. Focus on the primary activity of the main subject
3. If multiple people, classify the most prominent person
4. Provide confidence level (0-100%) based on clarity and certainty
5. Give a brief reasoning for your classification

OUTPUT FORMAT (JSON):
{
    "activity": "sit/walk/stand",
    "confidence": 85,
    "reasoning": "Brief explanation of why this classification was chosen",
    "details": {
        "pose_description": "Description of the observed pose/position",
        "context": "Environmental context that supports the classification"
    }
}

Now classify this new image following the same format:"""
        return prompt


    def classify_activity(self, image_path):
        """
        Classify activity in an image with confidence and structured output

        Args:
            image_path (str): Path to the image file

        Returns:
            dict: Structured classification result
        """
        # Create messages with example images for few-shot learning
        messages = []

        # Example images from your dataset
        example_sit = "/mnt/ext-data/qwen_vl/data/02-reading-chair-ideas-homebnc.jpg"
        example_walk = "/mnt/ext-data/qwen_vl/data/brisk-walking.jpg"
        # System prompt with examples
        messages.append({
            'role': 'user',
            'content': """You are an expert human activity classifier. Your task is to classify images into one of four categories: sit, walk, stand, or none.

CLASSIFICATION GUIDELINES:
- "sit": Person is seated (on chair, bench, ground, etc.)
- "walk": Person is walking, jogging, running, or in motion
- "stand": Person is standing upright, stationary
- "none": Multiple people doing different activities, OR no clear human activity visible

IMPORTANT RULES:
1. If there are multiple people doing DIFFERENT activities (e.g., one sitting, one standing), classify as "none"
2. If there are multiple people doing the SAME activity (e.g., all walking together), use that activity
3. If no people are clearly visible or identifiable, classify as "none"
4. Focus on the most prominent/clear activity when people are doing the same thing

Here are examples:"""
        })

        # Example 1: Sitting
        if os.path.exists(example_sit):
            messages.append({
                'role': 'user',
                'content': 'EXAMPLE 1 - Classify this image:',
                'images': [self.encode_image_to_base64(example_sit)]
            })

            messages.append({
                'role': 'assistant',
                'content': """{
    "activity": "sit",
    "confidence": 95,
    "reasoning": "Person is clearly seated in a chair in a reading position",
    "details": {
        "pose_description": "Seated posture with back against chair, relaxed position",
        "context": "Indoor reading environment with chair"
    }
}"""
            })

        # Example 2: Walking
        if os.path.exists(example_walk):
            messages.append({
                'role': 'user',
                'content': 'EXAMPLE 2 - Classify this image:',
                'images': [self.encode_image_to_base64(example_walk)]
            })

            messages.append({
                'role': 'assistant',
                'content': """{
    "activity": "walk",
    "confidence": 92,
    "reasoning": "Person is in active walking motion with clear gait pattern",
    "details": {
        "pose_description": "Active walking posture with leg movement and forward motion",
        "context": "Outdoor walking path or exercise environment"
    }
}"""
            })

        # Add example for "none" classification
        messages.append({
            'role': 'user',
            'content': 'EXAMPLE 3 - Multiple people with different activities:'
        })

        messages.append({
            'role': 'assistant',
            'content': """{
    "activity": "none",
    "confidence": 90,
    "reasoning": "Multiple people visible doing different activities - some sitting, others standing",
    "details": {
        "pose_description": "Mixed activities detected in the same image",
        "context": "Multiple people with conflicting activity classifications"
    }
}"""
        })

        # Add the target image to classify
        messages.append({
            'role': 'user',
            'content': 'Now classify this NEW image following the same JSON format. Remember: use "none" if multiple people are doing different activities:',
            'images': [self.encode_image_to_base64(image_path)]
        })

        try:
            response = chat(
                model=self.model_name,
                messages=messages
            )
            response_text = response['message']['content']

            # Try to parse JSON from response
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1

                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    # Fallback parsing
                    result = self._parse_text_response(response_text)
            except json.JSONDecodeError:
                result = self._parse_text_response(response_text)

            result['image_path'] = image_path
            result['raw_response'] = response_text
            return result

        except Exception as e:
            return {"error": f"Classification failed: {str(e)}", "image_path": image_path}

    def _parse_text_response(self, text):
        """
        Parse text response into structured format as fallback

        Args:
            text (str): Raw text response from model

        Returns:
            dict: Structured result
        """
        text_lower = text.lower()

        # Determine activity
        activity = "unknown"
        if "none" in text_lower:
            activity = "none"
        elif "sit" in text_lower:
            activity = "sit"
        elif "walk" in text_lower:
            activity = "walk"
        elif "stand" in text_lower:
            activity = "stand"

        return {
            "activity": activity,
            "confidence": 75,
            "reasoning": text[:200] + "..." if len(text) > 200 else text,
            "details": {
                "pose_description": "Extracted from text response",
                "context": "Text parsing fallback"
            }
        }

    def test_activity_classification(self):
        """
        Test the activity classification with sample images
        """
        test_images = [
            "/mnt/ext-data/qwen_vl/data/download.jpg", 
        
        ]

        print("=== Activity Classification Test ===")
        print("Using few-shot prompting with EXAMPLE IMAGES and structured output\n")

        results = []
        activity_counts = {"sit": 0, "walk": 0, "stand": 0, "none": 0, "error": 0}

        for i, image_path in enumerate(test_images, 1):
            import os
            image_name = os.path.basename(image_path)
            print(f"Processing {i}/{len(test_images)}: {image_name}")

            result = self.classify_activity(image_path)
            results.append(result)

            if "error" not in result:
                activity = result.get("activity", "unknown")
                confidence = result.get("confidence", 0)
                reasoning = result.get("reasoning", "No reasoning")

                print(f"  Result: {activity.upper()} ({confidence}%)")
                print(f"  Reasoning: {reasoning[:100]}...")

                if activity in activity_counts:
                    activity_counts[activity] += 1
                else:
                    activity_counts["error"] += 1
            else:
                print(f"  Error: {result['error']}")
                activity_counts["error"] += 1

            print()

        # Summary
        print("="*50)
        print("SUMMARY:")
        print("="*50)
        print(f"Total images: {len(test_images)}")
        print(f"Successful: {len(test_images) - activity_counts['error']}")
        print(f"Failed: {activity_counts['error']}")
        print(f"\nActivity Distribution:")
        for activity, count in activity_counts.items():
            if activity != "error":
                percentage = (count / len(test_images) * 100) if len(test_images) > 0 else 0
                print(f"  {activity.capitalize()}: {count} ({percentage:.1f}%)")

        return results

# Example usage
if __name__ == "__main__":
    # Initialize the model
    qwen_vl = QwenVLModel("qwen3-vl:8b")

    print("Testing Qwen VL model with Activity Classification...")

    # Test activity classification with your images
    qwen_vl.test_activity_classification()

    # Optional: Test individual image
    print("\n" + "="*50)
    print("INDIVIDUAL TEST EXAMPLE:")
    print("="*50)

    single_image = "/mnt/ext-data/qwen_vl/data/brisk-walking.jpg"
    result = qwen_vl.classify_activity(single_image)

    if "error" not in result:
        print(f"Activity: {result['activity'].upper()}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Reasoning: {result['reasoning']}")
        print(f"Details: {result['details']}")
    else:
        print(f"Error: {result['error']}")