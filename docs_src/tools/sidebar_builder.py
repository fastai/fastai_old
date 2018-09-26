def needs_slash(string):  
    "Checks if a line is 'title' and concatenates a slash if it is."
    slash =''
    if str(string) == 'title':    
        slash == '- '
    return slash
    
def write_line(data, text_file, slash, key, key2=None, key3=None):
    "Writes down a line of the sidebar."
    if key2 and key3:
        text_file.write(('\t\t'+str(slash)+str(key3)+str(': ')+str(data[key][key2][key3])+'\n').strip('\''))
    elif key2:
        text_file.write(('\t'+str(slash)+str(key2)+': '+str(data[key][key2])+'\n').strip('\''))
    else:
        text_file.write((str(key)+':\n').strip('\''))
        
def build_sidebar(data, file_name):
    "Builds a yaml sidebar from a dictionary of sections."
    with open(file_name+'.yml', "w") as text_file:
        for key in data:
            slash = needs_slash(key)
            write_line(data, text_file, slash, key)
            for key2 in data[key]:
                if type(data[key][key2]) == str:
                    slash = needs_slash(key2)
                    write_line(data, text_file, slash, key, key2)
                elif type(data[key][key2]) == dict:
                    slash = needs_slash(key2)
                    for key3 in data[key][key2]:
                        slash = needs_slash(key3)
                        write_line(data, text_file, slash, key, key2, key3)