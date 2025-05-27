from openai import OpenAI
import base64
import cv2


class OpenAI_API:
    def __init__(self):
        self.api_key = ""
        self.base_url = ""
        self.gpt_client = OpenAI(api_key=self.api_key, base_url = self.base_url)
        # self.gpt_client = OpenAI(api_key=self.api_key)
    
    def detection_refinement(self, image, object):
        base64_image = self.encode_image_from_array(image)
        if object.lower().startswith("couch"):
            # prompt =f"The couch must have at least three seat for person,the chair for one person is not couch. is there a/an {object} in the bbox of the given image? Please answer yes or no. "
            prompt =f''' onsider the reasonableness of the {object}'s appearance in the environment of the given iamge. is there a/an {object} in the contour line/bbox of the given image? if there is another couch in image is bigger than the couch in bbox, please answer also no. Please answer yes or no.
            1. The couch must have at least three seat for person,the chair for one person is not couch.
            2. The couch should in living room or bedroom, not in other rooms like kitchen or bathroom.'''
        elif object.lower().startswith("tv"):
            prompt = f'''onsider the reasonableness of the {object}'s appearance in the environment of the given iamge. is there a/an {object} in the contour line/bbox of the given image? Please answer yes or no. 
            1. Be careful not to mistake the picture frames and black Windows on the walls for televisions.
            2. The tv screen shuold be black, not white or other colors.
            3. There is no tv on a door or in toilet!
            4. only output yes or no.'''
        elif object.lower().startswith("chair"):
            prompt = f'''onsider the reasonableness of the {object}'s appearance in the environment of the given iamge. is there a/an {object} in the contour line/bbox of the given image? Please answer yes or no. Be careful not to mistake the couch for chair
            1. please attention: the chair only have one seat, but the couch have seats over one.
            2. only output yes or no.'''
        elif "bed" in object.lower():
            prompt = f'''onsider the reasonableness of the {object}'s appearance in the environment of the given iamge. is there a/an {object} in the contour line/bbox of the given image? Please answer yes or no.
            1. Please attention: the bed can only be in bedroom, not in other rooms like living room or bathroom and so on. If the environment is living room, there is no bed.
            2. please attention the environment of the image, if you are not sure, please answer no.
            2. If the detected object is on the edge of the image and you are not sure, please answer No.
            3. only output yes or no.'''        
        else:
            prompt = f"onsider the reasonableness of the {object}'s appearance in the environment of the given iamge. is there a/an {object} in the contour line/bbox of the given image? Please answer yes or no."
        
        completion = self.gpt_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    )
        answer = completion.choices[0].message.content
        return answer
    
    def detection_choise(self, image, object, color_string_list):
        base64_image = self.encode_image_from_array(image)
        if "couch" in object:
            prompt = f'''There are {len(color_string_list)} {object} in the image with {color_string_list} color contour line/bbox. you should choose only one object that best matches the feature of {object}. Please give the color of contour line/bbox of the chosen object. Only output one color in {color_string_list} of the contour line/bbox
            1. If there are some couches in this image, select the target with the most seats.
            2. Only output the color of the contour line/bbox.'''
        if "chair" in object:
            prompt = f'''There are {len(color_string_list)} {object} in the image with {color_string_list} color contour line/bbox. you should choose only one object that best matches the feature of {object}. Please give the color of contour line/bbox of the chosen object. Only output one color in {color_string_list} of the contour line/bbox
            1. please choose the chair which only has one seat, when there are some couches in the image. 
            2. Only output the color of the contour line/bbox.'''
        else:
            prompt = f"There are {len(color_string_list)} {object} in the image with {color_string_list} color contour line/bbox. you should choose only one object that best matches the feature of {object}. Please give the color of contour line/bbox of the chosen object. Only output one color in {color_string_list} of the contour line/bbox"
        completion = self.gpt_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    )
        print("prompt: ",prompt)
        answer = completion.choices[0].message.content
        return answer
    
    
    def encode_image_from_array(self, image):
        # 将 OpenCV BGR 图像转换为 JPEG 格式的字节流
        _, buffer = cv2.imencode(".jpg", image)
        # 将字节流转换为 Base64 字符串
        return base64.b64encode(buffer).decode("utf-8")