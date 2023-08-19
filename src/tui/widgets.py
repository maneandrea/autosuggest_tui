import curses
import curses.ascii
import re

import npyscreen as nps

from transformer import tokenizer


class MultiLineAuto(nps.MultiLineEdit):
    """A version of MultiLineEdit with an autocomplete feature"""

    WELCOME_TEXT = 'Start writing here...'

    def __init__(self, *args, suggester=None, **kwargs):
        # I need to set _old_value to WELCOME_TEXT otherwise __init__ invokes when_value_edited()
        self._old_value = self.WELCOME_TEXT
        self.suggester = suggester
        super(MultiLineAuto, self).__init__(value=self.WELCOME_TEXT, *args, **kwargs)
        self.untouched = True

    def when_value_edited(self):
        if self.untouched:
            self.color = 'DEFAULT'
            self.value = self.value.replace(self.WELCOME_TEXT, '')
            self.untouched = False
            self.handlers.update({curses.ascii.TAB: self.auto_complete})

    def set_up_handlers(self):
        super(MultiLineAuto, self).set_up_handlers()

        self.handlers.update({
            curses.ascii.TAB: self.auto_complete_first,
            curses.KEY_END: self.goto_end,
            curses.KEY_HOME: self.goto_home
        })

    def auto_complete_first(self, input):
        """Only in case TAB is pressed at the very beginning"""
        self.when_value_edited()
        self.auto_complete(input)

    def auto_complete(self, input):
        """Applies the autocomplete choice"""

        try:
            text = self.value[:self.cursor_position]
        except IndexError:
            text = self.value

        # choices contains a list of pairs (word, probability)
        choices = self.suggester.suggest(
            list(tokenizer.tokenize(text, self.suggester.CONTEXT_SIZE))
        )
        i = self.get_choice(choices)
        if i is not None:
            self.add_word(choices[i][0])

    def get_choice(self, values):
        """Shows a drop-down menu of autocomplete options and returns the user's choice"""

        # Display probabilities nicely
        to_word = lambda y: str(y).rstrip(' \n') if y else '<no suggestion>'
        values = [f'{to_word(v)} ({p*100:.2g}%)' for v, p in values]

        # This takes care of spawning the DropDown where the cursor is
        nps.Popup.DEFAULT_LINES = len(values) + 4
        nps.Popup.DEFAULT_COLUMNS = max(map(len, values)) + 5
        nps.Popup.SHOW_ATY = min(self.cursory + 2, self.height - nps.Popup.DEFAULT_LINES)
        nps.Popup.SHOW_ATX = min(self.cursorx + 3, self.width - nps.Popup.DEFAULT_COLUMNS)
        popup = nps.Popup(framed=True)
        popup.nextrelx = 1
        popup.nextrely = 1

        menu = popup.add_widget(DropDown, values=values)

        popup.display()
        return menu.choice()

    def goto_end(self, event):
        """Go to the end of the line"""
        current = self.value[self.cursor_position]
        while current != '\n':
            try:
                self.cursor_position += 1
                current = self.value[self.cursor_position]
            except IndexError:
                self.cursor_position -= 1
                break
        else:
            self.cursor_position -= 1

    def goto_home(self, event):
        """Go to the beginning of the line"""
        self.cursor_position -= self.cursorx

    def add_word(self, word):
        """Adds a word to the text taking care of carriage return"""

        try:
            if self.cursorx + len(word) == self.width - 2:
                to_add = word.rstrip(' \n\t') + '\n'
                to_move = len(to_add)
            elif self.cursorx + len(word) > self.width - 2:
                if re.match('\s+', word) or self.value[self.cursor_position - 1] in ' .,;!?:':
                    to_add = '\n' + word.lstrip(' \n\t')
                    to_move = len(to_add)
                else:
                    to_add = '-\n' + word
                    to_move = len(word) + 2
            else:
                to_add = word
                to_move = len(word)

            self.value = self.value[:self.cursor_position] + to_add + self.value[self.cursor_position:]
            self.cursor_position += to_move

        except IndexError:
            self.value += word
            self.cursor_position += len(word)


class DropDown(nps.MultiLine):
    """Multiline specialized for a drop-down menu"""

    def __init__(self, *args, **kwargs):
        new_kwargs = kwargs | {'return_exit': True, 'select_exit': True}
        super(DropDown, self).__init__(*args, **new_kwargs)

        self.handlers.update({
            curses.KEY_BACKSPACE: self.h_exit_escape,
            curses.ascii.DEL: self.h_exit_escape,
            curses.KEY_DC: self.h_exit_escape
        })

        self.value = 0

    def choice(self):
        """Returns the position of the chosen option or None if the autocomplete was aborted"""

        self.edit()
        if self.how_exited == nps.widget.EXITED_ESCAPE:
            return None
        else:
            return self.value
