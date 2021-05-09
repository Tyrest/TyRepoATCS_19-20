def fubar(beat, yeet=True, *sabers, **songs):
    print("beat", beat, type(beat))
    print("yeet", yeet, type(yeet))
    for saber in sabers:
        print("unnamed", saber, type(saber))
    for song in songs:
        print(song, songs[song], type(songs[song]))
    print("-------------------------------------------------------------------")

fubar("yes", False, 32,4668,78,678,69,hello=324,boi=234.25,y43="345",bad=True)
fubar("maybe",34,34,"EFOISH",34.34,hello=435,mayhb4=234,sdoifbs="idub",yes="No")
fubar("no")
