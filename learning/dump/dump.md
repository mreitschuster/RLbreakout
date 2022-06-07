 

# remarks

Have a look at your the command line tool nvidia-smi. It tells you how much of you GPU memory is used.If you cannot train a model you might be out of memory. Check here and if necessary close some proceeses. I like having it update continously with the watch command:<br>
<code>watch nvidia-smi</code><br>

Your GPU memory will restrict how many models you can train at the same time.
