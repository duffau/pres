
#!/usr/env bash
while inotifywait -e close_write slides.md 
do 
    make slides.html 
done