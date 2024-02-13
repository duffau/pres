---
title: Git from scratch
author: Christian Duffau-Rasmussen
date: 2023-02-13
title-slide-attributes:
    data-background-image: "./presentation/front-cover.png"
    data-background-size: 40%
    data-background-opacity: "0.1"
---

## Sources

- [https://git-scm.com/book Chapter 10 - Git Internals](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain)
- [https://github.blog/git-database-internals](https://github.blog/2022-08-29-gits-database-internals-i-packed-object-store/)
- [Git Internals - Scot Chacon (2008)](https://github.com/pluralsight/git-internals-pdf)

## Initialiazing the repo

```{.sh include=git-internals-pres.sh startFrom=7 endAt=8}
```

```{.bash include=git-internals-pres.sh startFrom=13 endAt=13}
```
. . . 

### ... that's to easy

```{.bash include=git-internals-pres.sh startFrom=16 endAt=16}
```

## Initialiazing the repo manually

```{.bash include=git-internals-pres.sh startFrom=20 endAt=24}
```

. . .

```{.bash include=git-internals-pres.sh startFrom=28 endAt=28}
```

. . . 

```{.bash include=git-internals-pres.sh startFrom=32 endAt=34}
```

... and  we see `git` is happy ... for now ...

## Adding a file

```{.bash include=git-internals-pres.sh startFrom=40 endAt=40}
```