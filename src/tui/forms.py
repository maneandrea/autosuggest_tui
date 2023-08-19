import npyscreen as nps
import curses

from tui import widgets


class MainForm(nps.Form):

    OK_BUTTON_TEXT = 'Exit'

    def __init__(self, suggester=None, **kwargs):
        self.suggester = suggester
        super(MainForm, self).__init__(**kwargs)

    def create(self):
         self.add(
            widgets.MultiLineAuto,
            color='LABEL',
            slow_scroll=True,
            suggester=self.suggester
        )

    def highlight_ok(self):
        """Set focus back to the ok button"""
        for n, w in enumerate(self._widgets__):
            if w == self.ok_button:
                self.editw = n
                self.display()
                return

    def set_up_exit_condition_handlers(self):
        super(MainForm, self).set_up_exit_condition_handlers()
        self.how_exited_handers.update({nps.widget.EXITED_ESCAPE: self.highlight_ok})

    def afterEditing(self):
        self.parentApp.setNextForm(None)

    def draw_title_and_help(self):
        """Overridden method just for the purpose of printing the title bold"""
        if self.name:
            _title = self.name[:(self.columns-4)]
            _title = ' ' + str(_title) + ' '
            self.add_line(0, 1,
                          _title,
                          self.make_attributes_list(_title, curses.A_BOLD),
                          self.columns-4)