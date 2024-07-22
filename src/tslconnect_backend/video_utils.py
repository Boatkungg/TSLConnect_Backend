import glob
import os
import time
import uuid

from moviepy.editor import VideoFileClip, concatenate_videoclips

words_dir = "data/words"

output_dir = "results/"


def make_video_from_words(words):
    video_files = [
        glob.glob(os.path.join(words_dir, word, "*.mp4"))[0] for word in words
    ]
    clips = [VideoFileClip(video_file) for video_file in video_files]
    final_clip = concatenate_videoclips(clips)

    os.makedirs(output_dir, exist_ok=True)
    video_name = f"{uuid.uuid4()}_{int(time.time())}.mp4"
    output_path = os.path.join(output_dir, video_name)
    
    final_clip.write_videofile(output_path, codec="libx264")
    return video_name
