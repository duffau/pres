.PHONY: .slides.html sim

reveal_flags = --mathjax \
		--highlight-style zenburn \
		--slide-level 3 \
		-V revealjs-url=https://unpkg.com/reveal.js@^4/ \
		-V theme=black \
		-V transition="linear" \
		-V slideNumber="'c/t'" \
		--css ./black-root.css \
		--bibliography ./bibliography.bib \
		--citeproc

sim:
	cd ./simulation; ./run.sh

slides.html:
	pandoc -t revealjs \
	  	-s slides.md \
		-o .slides.html \
		$(reveal_flags)
