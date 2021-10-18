% Git from scratch
% Christian Duffau-Rasmussen
% \today

## Initialiazing the repo

```{.sh include=git-internals-pres.sh snippet=make-myrepo}
```

```{.bash include=git-internals-pres.sh snippet=git-init}
```
. . . 

### ... that's to easy

```{.bash include=git-internals-pres.sh snippet=git-rm}
```

## Initialiazing the repo manually

```{.bash include=git-internals-pres.sh snippet=git-init-manual}
```

. . .

```{.bash include=git-internals-pres.sh snippet=git-status}
```

. . . 

```{.bash include=git-internals-pres.sh snippet=git-status-out}
```

... and  we see `git` is happy ... for now ...

## Adding a file

```{.bash include=git-internals-pres.sh snippet=make-first-file}
```