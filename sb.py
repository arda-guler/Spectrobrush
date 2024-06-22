import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import resample
from PIL import Image
import sys

helptext =\
"""
Spectrobrush

Author(s):
arda-guler (@ GitHub)

Paints an audio spectrum using image data.
A command line program.
Arguments:

-i: Input image file
-o: Pure sound output filename
-m: Merged output base filename
-om: Merged output filename
-d: Generated sound duration
-sr: Audio sample rate
-st: Merge start time
-fl: Low frequency limit
-fh: High frequency limit

Spectrobrush is licensed under MIT License.
See https://github.com/arda-guler/spectrobrush
for details.
"""

def image_to_brightness_array(image_path):
    with Image.open(image_path) as img:
        grayscale_img = img.convert('L')
        width, height = grayscale_img.size
        grayscale_array = np.array(grayscale_img)
        
        brightness_array = grayscale_array / 255.0
        brightness_array = brightness_array.T
        brightness_array = brightness_array.tolist()
        
        return brightness_array

def brightness_array_to_sound(brightness_array, duration, min_freq, max_freq, sample_rate=44100):
    num_columns = len(brightness_array)
    num_rows = len(brightness_array[0])
    time_per_column = duration / num_columns

    # log scale
    frequencies = np.linspace(min_freq, max_freq, num=num_rows)[::-1]

    sound_data = np.zeros(int(duration * sample_rate), dtype=np.float32)

    for i in range(num_columns):
        start_sample = int(i * time_per_column * sample_rate)
        end_sample = int((i + 1) * time_per_column * sample_rate)

        for j in range(num_rows):
            amplitude = brightness_array[i][j]
            
            if amplitude > 0:
                t = np.linspace(0, time_per_column, end_sample - start_sample, endpoint=False)
                wave = amplitude * np.sin(2 * np.pi * frequencies[j] * t)
                sound_data[start_sample:end_sample] += wave

    # normalize
    max_val = np.max(np.abs(sound_data))
    if max_val > 0:
        sound_data = sound_data / max_val

    return sound_data

def save_sound_file(sound_data, sample_rate, output_file):
    int_data = np.int16(sound_data * 32767) # not so much of a magic number now, is it?
    write(output_file, sample_rate, int_data)

def merge_sounds(existing_sound_file, generated_sound_data, sample_rate, output_file, start_time):
    existing_sample_rate, existing_sound_data = read(existing_sound_file)
    
    # handle stereo and mono
    if existing_sound_data.ndim == 2:
        is_stereo = True
        num_channels = existing_sound_data.shape[1]
    else:
        is_stereo = False
        num_channels = 1
        existing_sound_data = existing_sound_data[:, np.newaxis]
    
    # resample generated sound if the sample rates do not match
    if existing_sample_rate != sample_rate:
        num_samples = int(len(generated_sound_data) * existing_sample_rate / sample_rate)
        generated_sound_data = resample(generated_sound_data, num_samples)
        sample_rate = existing_sample_rate
    
    # expand generated sound to match the number of channels in the existing file
    generated_sound_data = np.tile(generated_sound_data[:, np.newaxis], (1, num_channels))
    
    start_sample = int(start_time * sample_rate)
    
    # length compatibility
    total_length = max(start_sample + len(generated_sound_data), len(existing_sound_data))
    existing_sound_data = np.pad(existing_sound_data, ((0, total_length - len(existing_sound_data)), (0, 0)), 'constant')
    generated_sound_data = np.pad(generated_sound_data, ((0, total_length - (start_sample + len(generated_sound_data))), (0, 0)), 'constant')
    
    merged_sound_data = np.zeros((total_length, num_channels), dtype=np.float32)
    merged_sound_data[:len(existing_sound_data), :] += existing_sound_data
    
    merged_sound_data[start_sample:start_sample + len(generated_sound_data), :] += generated_sound_data[:len(generated_sound_data), :]
    
    # normalize!
    max_val = np.max(np.abs(merged_sound_data))
    if max_val > 0:
        merged_sound_data = merged_sound_data / max_val
    
    int_data = np.int16(merged_sound_data * 32767)
    
    write(output_file, sample_rate, int_data)

def main(sys_args):
    image_input_filename = "test.png"
    pure_output_filename = "pure.wav"
    merge_base_filename = ""
    merge_output_filename = "merged.wav"

    freq_low = 200
    freq_high = 10000

    duration = 5
    sample_rate = 44100
    start_time = 0

    argtype = "-i"
    for i, arg in enumerate(sys_args):
        if not i == 0:
            if arg.startswith("-"):
                if arg == "-h" or arg == "--help":
                    print(helptext)
                    return
                else:
                    argtype = arg
            else:
                if argtype == "-i":
                    image_input_filename = arg
                elif argtype == "-o":
                    pure_output_filename = arg
                elif argtype == "-m":
                    merge_base_filename = arg
                elif argtype == "-om":
                    merge_output_filename = arg
                elif argtype == "-fl":
                    freq_low = int(arg)
                elif argtype == "-fh":
                    freq_high = int(arg)
                elif argtype == "-d":
                    duration = float(arg)
                elif argtype == "-sr":
                    sample_rate = int(arg)
                elif argtype == "-st":
                    start_time = float(arg)

    # sanitize input
    if not pure_output_filename.endswith(".wav"):
        pure_output_filename = pure_output_filename + ".wav"

    if merge_base_filename and not merge_base_filename.endswith(".wav"):
        merge_base_filename = merge_base_filename + ".wav"

    if not merge_output_filename.endswith(".wav"):
        merge_output_filename = merge_output_filename + ".wav"
    
    brightness_array = image_to_brightness_array(image_input_filename)
    generated_sound_data = brightness_array_to_sound(brightness_array, duration, freq_low, freq_high, sample_rate)
    save_sound_file(generated_sound_data, sample_rate, pure_output_filename)

    if merge_base_filename:
        merge_sounds(merge_base_filename, generated_sound_data, sample_rate, merge_output_filename, start_time)

if __name__ == "__main__":
    main(sys.argv)
