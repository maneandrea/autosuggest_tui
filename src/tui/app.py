import npyscreen as nps
import curses

from tui import forms


class AutosuggestApp(nps.NPSAppManaged):

    def __init__(self, suggester):
        super(AutosuggestApp, self).__init__()
        self.suggester = suggester

    def onStart(self):
        self.addForm('MAIN', forms.MainForm, name='Autosuggestion', suggester=self.suggester)
        curses.set_escdelay(25)
