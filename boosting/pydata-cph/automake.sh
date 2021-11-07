
#!/usr/env bash
while inotifywait -e close_write --format "%f %T" --timefmt "%T" --exclude .slides.html -r ./ 
do 
    make slides.html 
done