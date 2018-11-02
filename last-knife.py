import time
import itertools

import numpy as np

import cv2 as cv
from mss import mss
from PIL import Image

import pyautogui

from uncertainties import ufloat
import uncertainties.unumpy as unp


SHOW_PREVIEW = True
GAME_CENTER = (250, 150)

z = 2


class Timeout:
    def __init__(self, delay):
        self.delay = delay
        self.last = 0

    def __bool__(self):
        return time.time() - self.last > self.delay

    def reset(self):
        self.last = time.time()


size = np.array([350, 340])
center = np.array([164, 177])

sample_points = 512
history_length = 16

target_frame_rate = 60
level_timeout = Timeout(0.7)
print_timeout = Timeout(0.2)

lookback_time = 0.08

target_angle = 90
base_width = 12.5

click_position = np.array([315, 760])
click_timeout = Timeout(0.05)

speed_cap = 300
click_latency = 0.11


target_color = np.array([0, 0, 255], dtype=np.uint8)
prediction_color = np.array([0, 255, 0], dtype=np.uint8)
activation_color = np.array([255, 0, 0], dtype=np.uint8)


bounding_box = {
    'top': GAME_CENTER[0],
    'left': GAME_CENTER[1],
    'width': size[0],
    'height': size[1]
}

def circle(radius):
    y, x = np.mgrid[:size[1], :size[0]]
    return np.hypot(x - center[0], y - center[1]) < radius

def polar(radius, angle):
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    return np.array([x, y])

def correlate_spectrum(x, y):
    return np.fft.ifft(x * y.conj()).real

def autocorrelate_spectrum(spectrum):
    return np.fft.ifft(spectrum[:-1] * spectrum[1:].conj()).real

def average(measurements):
    #mean = np.mean(measurements)
    mean = np.median(measurements)
    #uncertainty = np.std(measurements, ddof=1) / np.sqrt(len(measurements)) #Standard error of the mean
    #uncertainty = np.max(np.abs(measurements - mean)) # Maximum absolute deviation
    uncertainty = np.mean(np.abs(measurements - mean)) # Mean absolute deviation
    #uncertainty = np.sqrt(np.mean(np.square(measurements - mean))) # RMS
    return ufloat(mean, uncertainty)

def degrees_to_bins(degrees):
    return int(round(degrees * sample_points/360))

sample_radians = np.linspace(0, 2*np.pi, sample_points)
sample_degrees = sample_radians * 180/np.pi

activation_mask = np.rint(polar(90, sample_radians)).astype(int)
sample_x, sample_y = np.rint(polar(149, sample_radians)).astype(int)


history = np.zeros((history_length, sample_points), dtype=complex)
frame_time = np.zeros(history_length)
speed_history = np.zeros(history_length)

state_history = []

prediction_shift = 0
fps = ufloat(0, 0)
clear_frames = 0
try:
    with mss() as sct:
        for frame_number in itertools.count():
            # Capture frame and record timestamp
            frame_time[1:] = frame_time[:-1]
            frame_time[0] = time.time()
            frame = np.array(sct.grab(bounding_box))

            # Compute FPS
            frame_lookback = int(min(max(3, np.ceil(fps.n * lookback_time)), history_length-1))
            spf_history = -np.diff(frame_time[:frame_lookback])

            spf = average(spf_history)
            fps = 1/(spf + 1e-5)


            ## Feature Extraction
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            #cv.imshow('HSV Space', hsv)

            mask = ~cv.inRange(hsv,
                np.array([80, 210, 50]),
                np.array([110, 255, 150])
            )
            #cv.imshow('Mask', mask)

            kernel = np.ones((2, 2), dtype=np.uint8)
            cleaned = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

            preview = cv.cvtColor(cleaned, cv.COLOR_GRAY2RGB)
            #preview[activation_mask[1], activation_mask[0]] = activation_color

            
            ## Interpretation
            active = True #np.all(cleaned[activation_mask[1], activation_mask[0]])

            if active:
                # Sample values in relevant area
                values = cleaned[sample_y, sample_x].astype(np.bool)
                history[1:] = history[:-1]
                history[0] = np.fft.fft(values)

                # Estimate rotation speed
                #correlation = autocorrelate_spectrum(history[:frame_lookback]) / sample_points
                correlation = correlate_spectrum(history[0], history[frame_lookback]) / sample_points

                shift = sample_points // 2
                shifted_correlation = np.roll(correlation, shift=shift)#, axis=1)
                shifted_correlation -= 1e-2 * np.abs(np.arange(sample_points) - shift) / sample_points

                #speeds = 360/sample_points * (np.argmax(shifted_correlation, axis=1) - speed_cap_bins) * fps_history
                instantaneous_speed =  360/sample_points * (np.argmax(shifted_correlation) - shift) * fps.n/frame_lookback

                speed_history[1:] = speed_history[:-1]
                speed_history[0] = instantaneous_speed
                speed = average(speed_history[:frame_lookback])

                if speed.s > 400 or (abs(speed.n) + z*speed.s) < 0.01:
                    level_timeout.reset()

                # Estimate click latency
                total_latency = ufloat(click_latency, spf.n + z*spf.s)

                # Predict future state
                prediction_shift = speed * total_latency
                prediction = np.roll(values, shift=degrees_to_bins(prediction_shift.n))

                # Determine whether state will be clear
                window_radius = base_width/2 + z*prediction_shift.s
                target_indices = np.abs(sample_degrees - target_angle) < window_radius
                window_width = 2 * window_radius

                clear = not np.any(prediction & target_indices)

                preview[sample_y, sample_x] = np.outer(target_indices, target_color) | np.outer(prediction, prediction_color)

                if (clear
                        and level_timeout
                        and click_timeout
                        and abs(speed.n) + z * speed.s < speed_cap):
                    clear_frames += 1
                else:
                    clear_frames = 0

                if clear_frames > frame_lookback:
                    pyautogui.click(*click_position)
                    click_timeout.reset()
                    clear_frames = 0
                    state_history.append([fps, speed, window_width, preview])


                if SHOW_PREVIEW:
                    cv.imshow('Live Feed', preview)

                if print_timeout:
                    print('FPS: {:5.1f} +- {:5.1f}, Speed: {:6.1f} +- {:5.1f}, Window: {:6.1f}'.format(
                        fps.n, fps.s,
                        speed.n, speed.s,
                        window_width
                    ))
                    print_timeout.reset()


            #history_shift = np.arange(history_length) * speed.n * spf.n + prediction_shift.n
            #rolled_history = np.array([np.roll(frame, shift=degrees_to_bins(shift)) for frame, shift in zip(np.fft.ifft(history).real, history_shift)])
            #cv.imshow('History', (rolled_history * 255).astype(np.uint8))

            if cv.waitKey(25) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break

            time.sleep(max(0, 1/target_frame_rate - (time.time() - frame_time[0])))
finally:
    fps, speed, window_width, preview = state_history[-1]
    image = Image.fromarray(preview[:,:,::-1])
    image.save('final-state.png')
    
    print()
    print('----- Final Click -----')
    print('FPS: {:5.1f} +- {:5.1f}, Speed: {:6.1f} +- {:5.1f}, Window: {:6.1f}'.format(
                        fps.n, fps.s,
                        speed.n, speed.s,
                        window_width
                    ))
