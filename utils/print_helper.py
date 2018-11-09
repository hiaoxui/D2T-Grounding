from colorama import Style, Fore


def use_style(string, fore=''):

    if fore.lower() == 'green':
        return f'{Fore.GREEN}%s{Style.RESET_ALL}' % string

    if fore.lower() == 'red':
        return f'{Fore.RED}%s{Style.RESET_ALL}' % string

    if fore.lower() == 'blue':
        return f'{Fore.BLUE}%s{Style.RESET_ALL}' % string

    return '[[[%s]]]' % string


def print_sep(sep='=', n=50):
    print(sep * n)
