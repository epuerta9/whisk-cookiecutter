import os
import sys

def remove_file(filepath):
    os.remove(os.path.join(os.getcwd(), filepath))

def remove_dir(dirpath):
    os.rmdir(os.path.join(os.getcwd(), dirpath))

def main():
    if '{{ cookiecutter.include_query_handler }}' != 'y':
        remove_file('src/{{ cookiecutter.package_name }}/handlers/query.py')

    if '{{ cookiecutter.include_storage_handler }}' != 'y':
        remove_file('src/{{ cookiecutter.package_name }}/handlers/storage.py')

    if '{{ cookiecutter.include_embed_handler }}' != 'y':
        remove_file('src/{{ cookiecutter.package_name }}/handlers/embed.py')

if __name__ == '__main__':
    main() 