while inotifywait slides.md; do
    cd .. && make slides.html
done
