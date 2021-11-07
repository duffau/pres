.PHONY: .slides.html sim

sim:
	cd ./simulation; ./run.sh

slides.html:
	pandoc -t revealjs \
	  	-s slides.md \
		-o .slides.html \
		--mathjax \
		--highlight-style zenburn \
		--slide-level 3 \
		-V revealjs-url=https://unpkg.com/reveal.js@^4/ \
		-V theme=black \
		-V transition="linear" \
		-V slideNumber="'c/t'" \
		--css ../black-root.css \
		--incremental \
		--bibliography ../bibliography.bib \
		--citeproc
