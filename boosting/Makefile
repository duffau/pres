.PHONY:slides.html

slides.html:
	pandoc \
		-t revealjs \
	  	-s slides.md \
		-o slides.html \
		--mathjax \
		--highlight-style zenburn \
		-V revealjs-url=https://unpkg.com/reveal.js@^4/ \
		-V theme=black \
		-V transition="linear" \
		-V slideNumber="'c/t'" \
		--css black-root.css \
		--incremental \
		--bibliography bibliography.bib \
		--citeproc 