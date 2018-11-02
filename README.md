# last-knife-bot
Desktop AI for the "Last Knife" messenger game.

## Usage
### Basic Setup
1. Open the "Last Knife" messenger game in any browser,
and start the first level.
2. Note the pixel coordinates of the center of the spinning wheel.
   Alternatively, guess the coordinates and iterate steps 3 and 4 until a good
   approximation is found.
3. Change the `GAME_CENTER` variable to the coordinates recorded in step 2.
4. Run `last-knife.py`.

### Interpreting the Preview Window
![Screenshot of bot during gameplay.](https://raw.githubusercontent.com/wgxli/last-knife-bot/master/screenshot.png)
* The white area represents the processed foreground from the game's video feed.
* The green arcs show the bot's prediction for where any obstacles will be
  upon clicking.
* The red arc shows the bot's uncertainty.
  If any of the predicted obstacle positions (green arcs) overlap with
  the red arc, the bot will not click.

## Requirements
Currently supports only Python 3.
Requires the following libraries:
* numpy
* pyautogui
* PIL
* uncertainties
* matplotlib
* cv2
* mss

## Troubleshooting
* The FPS is limited by the live preview window.
  Setting `SHOW_PREVIEW = False` may increase performance.
* If the bot makes too many errors, try increasing the variable `z`,
  which represents the bot's confidence threshold.
