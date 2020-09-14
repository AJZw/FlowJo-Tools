##############################################################################     ##    ######
#    A.J. Zwijnenburg                   2020-09-24           v1.4                 #  #      ##
#    Copyright (C) 2020 - AJ Zwijnenburg          GPLv3 license                  ######   ##
##############################################################################  ##    ## ######

## Copyright notice ##########################################################
# FlowJo Tools provides a python API into FlowJo's .wsp files.
# Copyright (C) 2020 AJ Zwijnenburg
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
##############################################################################

"""
Objects for reading and manipulation of Flowjo v10 Compensation Matrix (.mtx) files

:class: Matrix
Basic Matrix class

:class: CompensationMatrix
A compensation matrix representing matrix

:class: MTX
A class for the opening, saving, and manipulation of FlowJo .mtx compensation matrix files

"""

from __future__ import annotations
from typing import Union, Tuple, List
from copy import deepcopy

import os.path
from lxml import etree

class Matrix:
    """
    A basic matrix class with row/column metadata
    """
    def __init__(self, nrow: int=5, ncol: int=5, default: Union[None, int, float, bool]=None):
        self._row_count = nrow
        self._col_count = ncol

        self._matrix = [default]*(self._row_count*self._col_count)

        self._row_names = [None]*self._row_count
        self._col_names = [None]*self._col_count

    @property
    def row_count(self) -> int:
        return self._row_count
    
    @property
    def col_count(self) -> int:
        return self._col_count

    @property
    def row_names(self) -> List[Union[None, str]]:
        return self._row_names

    @row_names.setter
    def row_names(self, value: List[str]) -> None:
        """
        Setter for row_names
        """
        if self._row_count <= 0:
            raise ValueError("matrix doesnt contain any rows")
        
        if len(value) != self._row_count:
            raise ValueError("cannot replace row names with less/more names then matrix rows")

        # Check if all rownames are unique
        if len(value) != len(set(value)):
            raise ValueError("row names have to be unique")

        for i, row in enumerate(value):
            if row is None:
                self._row_names[i] = row
            else:
                self._row_names[i] = str(row)

    @property
    def col_names(self) -> List[Union[None, str]]:
        return self._col_names

    @col_names.setter
    def col_names(self, value: List[str]) -> None:
        """
        Setter for col_names
        """
        if self._col_count <= 0:
            raise ValueError("matrix doesnt contain any columns")
        
        if len(value) != self._col_count:
            raise ValueError("cannot replace column names with less/more names then matrix columns")

        # Check if all rownames are unique
        if len(value) != len(set(value)):
            raise ValueError("column names have to be unique")

        for i, col in enumerate(value):
            if col is None:
                self._col_names[i] = col
            else:
                self._col_names[i] = str(col)

    def row(self, identifier: Union[int, str]) -> Matrix:
        """
        Returns the matrix row specified by the id.
            :param identifier: an index or row name that identifies the row
            :returns: a Matrix of the row
        """
        if isinstance(identifier, int):
            index = self.__check_row_index(identifier)
        elif isinstance(identifier, str):
            index = self.__check_row_name(identifier)
        else:
            raise TypeError("invalid identifier type")

        matrix_index_start = index * self._col_count 
        matrix_index_end = (index + 1) * self._col_count

        output = Matrix(0, self._col_count)
        output.append_row(self._row_names[index], self._matrix[matrix_index_start:matrix_index_end])
        output.col_names = self._col_names

        return output

    def emplace_row(self, identifier: Union[int, str], data: list) -> None:
        """
        Sets a row to data
        """
        if isinstance(identifier, int):
            index = self.__check_row_index(identifier)
        elif isinstance(identifier, str):
            index = self.__check_row_name(identifier)
        else:
            raise TypeError("invalid identifier type")

        self.__check_row_input_list(data)

        matrix_index_start = index * self._col_count 
        matrix_index_end = (index + 1) * self._col_count

        self._matrix[matrix_index_start:matrix_index_end] = data

    def append_row(self, identifier: Union[None, str,], data: list) -> None:
        """
        Appends a row of data to the matrix data
        """
        # First check if identifier already exists
        if identifier:
            for name in self._row_names:
                if name == str(identifier):
                    raise KeyError("row key already exists, each row key must be unique")

        self.__check_row_input_list(data)

        self._row_names.append(identifier)
        self._row_count += 1
        self._matrix.extend(data)

    def erase_row(self, identifier: Union[int, str]) -> None:
        """
        Removes a row from the matrix
        """
        if isinstance(identifier, int):
            index = self.__check_row_index(identifier)
        elif isinstance(identifier, str):
            index = self.__check_row_name(identifier)
        else:
            raise TypeError("invalid identifier type")

        matrix_index_start = index * self._col_count 
        matrix_index_end = (index + 1) * self._col_count

        new_matrix = self._matrix[:matrix_index_start]
        new_matrix.extend(self._matrix[matrix_index_end:])
        
        self._matrix = new_matrix
        self._row_count -= 1
        self._row_names.pop(index)

    def col(self, identifier: Union[int, str]) -> Matrix:
        """
        Returns the matrix column specified by the id.
            :param identifier: an index or column name that identifies the column
            :returns: a Matrix of the column
        """
        if isinstance(identifier, int):
            index = self.__check_col_index(identifier)
        elif isinstance(identifier, str):
            index = self.__check_col_name(identifier)
        else:
            raise TypeError("invalid identifier type")

        column_data = []
        for row in range(0, self._row_count):
            col_index = (row * self._col_count) + index
            column_data.append(self._matrix[col_index])

        output = Matrix(self._row_count, 0)
        output.append_col(self._col_names[index], column_data)
        output.row_names = self._row_names

        return output

    def emplace_col(self, identifier: Union[int, str], data: list) -> None:
        """
        Sets a column to data
        """
        if isinstance(identifier, int):
            index = self.__check_col_index(identifier)
        elif isinstance(identifier, str):
            index = self.__check_col_name(identifier)
        else:
            raise TypeError("invalid identifier type")

        self.__check_col_input_list(data)

        for i, value in enumerate(data):
            matrix_index = self._col_count * i + index
            self._matrix[matrix_index] = value

    def append_col(self, identifier: Union[None, str], data: list) -> None:
        """
        Appends a column of data to the matrix data
        """
        # First check if identifier already exists
        if identifier:
            for name in self._col_names:
                if name == str(identifier):
                    raise KeyError("column key already exists, each column key must be unique")

        self.__check_col_input_list(data)

        # To add columns we need to grow and move values around
        # First push the necessary size on to the end
        append_size = self._row_count
        self._matrix.extend([None]*append_size)

        # Now move parts around and add value
        for index in reversed(range(0, self._row_count)):
            old_start = self._col_count * index
            old_end = self._col_count * (index + 1)

            new_start = (self._col_count + 1) * index
            new_end = (self._col_count + 1) * (index + 1) -1

            self._matrix[new_start:new_end] = self._matrix[old_start:old_end]
            self._matrix[new_end] = data[index]

        self._col_count += 1
        self._col_names.append(identifier)
    
    def erase_col(self, identifier: Union[int, str]) -> None:
        """
        Removes a column from the matrix
        """
        if isinstance(identifier, int):
            index = self.__check_col_index(identifier)
        elif isinstance(identifier, str):
            index = self.__check_col_name(identifier)
        else:
            raise TypeError("invalid identifier type")

        # First move the entrees, opening space for deletion at the end of the list
        for row in range(0, self._row_count):
            start_old = row * self._col_count
            start_old_index = start_old + index
            end_old_index = start_old_index + 1
            end_old = (row + 1) * self._col_count

            start_new = row * (self._col_count - 1)
            start_new_index = start_new + index
            end_new = (row + 1) * (self._col_count - 1)

            self._matrix[start_new:start_new_index] = self._matrix[start_old:start_old_index]
            self._matrix[start_new_index:end_new] = self._matrix[end_old_index:end_old]

        # Remove leftover entrees
        self._matrix = self._matrix[:-self._row_count]
        
        self._col_count -= 1
        self._col_names.pop(index)

    def append_matrix_row(self, other: Matrix) -> None:
        """
        Appends a matrix as new rows.
        """
        if self._col_names != other._col_names:
            raise ValueError("unequal column names. Column names must be equal for appending")

        for row_name in other.row_names:
            if row_name in self.row_names:
                raise ValueError("row name already defined in matrix")

        self.__check_row_input_list(other._matrix, other.row_count)

        self._row_names.extend(other._row_names)
        self._row_count += other._row_count

        self._matrix.extend(other._matrix)

    def append_matrix_col(self, other: Matrix) -> None:
        """
        Appends a matrix as new columns.
        """
        if self._row_names != other._row_names:
            raise ValueError("unequal row names. Row names must be equal for appending")

        for col_name in other.col_names:
            if col_name in self.col_names:
                raise ValueError("column name already defined in matrix")

        self.__check_col_input_list(other._matrix, other.col_count)

        # To add columns we need to grow and move values around
        # First push the necessary size on to the end
        append_size = self._row_count * other.col_count

        self._matrix.extend([None]*append_size)
        for index in reversed(range(0, self._row_count)):
            old_start = self._col_count * index
            old_end = self._col_count * (index + 1)

            new_start = (self._col_count + other.col_count) * index
            new_end = (self._col_count + other.col_count) * (index + 1) - other.col_count

            self._matrix[new_start:new_end] = self._matrix[old_start:old_end]
            self._matrix[new_end:new_end + other.col_count] = other._matrix[index*other.col_count: (index+1)*other.col_count]

        self._col_names.extend(other._col_names)
        self._col_count += other._col_count

    def order_row(self, order: List[str]) -> None:
        """
        Orders the matrix row to the given order.
            :raises ValueError: if order is incomplete
        """
        # Check order vs row names and the other way around
        raise NotImplementedError

    def __check_row_input_list(self, data: list, rows: int = 1) -> None:
        """
        Checks an input list for correct typing
            :param data: the data to check
            :param rows: the amount of rows the data will use
            :raises ValueError: if the typing is incorrect
        """
        # Check typing
        if self._matrix and self._matrix[0] != None:
            if not all(isinstance(x, type(self._matrix[0])) for x in data):
                raise ValueError("object type contained in data list must be identical to object type contained in matrix")
        elif data:
            if not all(isinstance(x, type(data[0])) for x in data):
                raise ValueError("a matrix can only contain one object type")
        
        # Check if length fits in matrix
        if len(data) != self._col_count * rows:
            raise ValueError("data length doesnt fit in a Matrix row")
    
    def __check_col_input_list(self, data: list, cols: int = 1) -> None:
        """
        Checks an input list for correct typing
            :param data: the data to check
            :param rows: the amount of columns the data will use
            :raises ValueError: if the typing is incorrect
        """
        # Check typing
        if self._matrix and self._matrix[0] != None:
            if not all(isinstance(x, type(self._matrix[0])) for x in data):
                raise ValueError("object type contained in data list must be identical to object type contained in matrix")
        elif data:
            if not all(isinstance(x, type(data[0])) for x in data):
                raise ValueError("a matrix can only contain one object type")
        
        # Check if length fits in matrix
        if len(data) != self._row_count * cols:
            raise ValueError("data length doesnt fit in a Matrix column")

    def __check_row_index(self, index: int) -> int:
        """
        Checks whether the row index is valid, and returns the positive index
            :raises IndexError: if index is out of range
            :returns: the positive row index
        """
        if index >= self._row_count:
            raise IndexError(f"matrix row index '{index}' out of range")
        
        if index < 0:
            index = self._row_count + index

            if index < 0:
                raise IndexError(f"matrix row index '{index}' out of range")
        
        return index
    
    def __check_col_index(self, index: int) -> int:
        """
        Checks whether the column index is valid, and returns the positive index
            :raises IndexError: if index is out of range
            :returns: the positive row index
        """
        if index >= self._col_count:
            raise IndexError(f"matrix column index '{index}' out of range")
        
        if index < 0:
            index = self._col_count + index

            if index < 0:
                raise IndexError(f"matrix column index '{index}' out of range")
        
        return index
    
    def __check_row_name(self, name: str) -> int:
        """
        Checks whether a specified row name exists and returns the row index
            :raises KeyError: if row name is not found
        """
        is_identifier = False
        for index, row_name in enumerate(self._row_names):
            if row_name == name:
                is_identifier = True
                break
        
        if not is_identifier:
            raise KeyError("matrix row name not found")
            
        return index
    
    def __check_col_name(self, name: str) -> int:
        """
        Checks whether a specified column name exists and returns the column index
            :raises KeyError: if column name is not found
        """
        is_identifier = False
        for index, col_name in enumerate(self._col_names):
            if col_name == name:
                is_identifier = True
                break
        
        if not is_identifier:
            raise KeyError("matrix column name not found")
            
        return index

    def __repr__(self) -> str:
        return f"Matrix({self._row_count}x{self._col_count})"

    def __str__(self) -> str:
        # make a nice output
        output = "\t"
        for name in self._col_names:
            output += f"{str(name)}:\t"

        for i, name in enumerate(self._row_names):
            output += f"\n{str(name)}:\t"

            start = i * self._col_count
            end = (i + 1) * self._col_count
            for value in self._matrix[start:end]:
                output += f"{str(value)}\t"
        
        return output

    def __getitem__(self, key: Union[slice, int]) -> Matrix:
        if isinstance(key, slice):
            index_start = self.__check_row_index(key.start)
            try:
                index_stop = self.__check_row_index(key.stop -1)
            except IndexError:
                index_stop = self._row_count -1

            if not key.step:
                step = 1
            else:
                step = key.step
            
            output = Matrix(0, self._col_count)
            output.col_names = self._col_names
            for index in range(index_start, index_stop + 1, step):
                matrix_start = index * self._col_count
                matrix_end = (index + 1) * self._col_count

                output.append_row(self._row_names[index], self._matrix[matrix_start:matrix_end])
            
            return output

        elif isinstance(key, int) or isinstance(key, str):
            return self.row(key)
        else:
            raise TypeError("invalid argument type")

    def __add__(self, other: Union[float, int, bool, Matrix]) -> Matrix:
        """
        Matrix addition, accepts all non-str primitive types, and Matrix type
        """
        matrix = deepcopy(self)
        matrix += other
        return matrix

    def __iadd__(self, other: Union[float, int, bool, Matrix]) -> None:
        """
        Matrix addition, accepts all non-str primitive types, and Matrix type
        """
        if isinstance(other, (bool, int, float)):
            for item in self._matrix:
                item += other
        
        elif isinstance(other, Matrix):
            # Now do a matrix add, we will implement two options
            # Option 1: the other Matrix row and columns are non named
            if all(name is None for name in other._row_names):
                if not all(name is None for name in other._col_names):
                    raise ValueError("For index based matrix addition, both row and column names must be undefined (None)")

                if self._row_count != other._row_count or self._col_count != other._col_count:
                    raise ValueError("Only matrixes of equal row and column size can be added based on their index")

                for i in range(0, len(self._matrix)):
                    self._matrix[i] += other._matrix[i]

                return self

            # Option 2: the Matrix row and columns are all named and will be added based on name
            else:
                if any(name is None for name in other._row_names) or any(name is None for name in other._col_names):
                    raise ValueError("For name based addition, all row and column names of the added matrix must be defined (NOT None)")

                for i, row_name in enumerate(other._row_names):
                    # Find index of row name
                    try:
                        index_row = self._row_names.index(row_name)
                    except ValueError:
                        continue

                    # Find index of column name
                    for j, col_name in enumerate(other._col_names):
                        try:
                            index_col = self._col_names.index(col_name)
                        except ValueError:
                            continue

                        index_self = (index_row * self._col_count) + index_col
                        index_other = (i * other._col_count) + j
                        self._matrix[index_self] += other._matrix[index_other]

                return self
        
        else:
            raise ValueError("Can only add objects of the following type to a matrix: bool, int, float, Matrix")

    def __sub__(self, other: Union[float, int, bool, Matrix]) -> Matrix:
        """
        Matrix substraction, accepts all non-str primitive types, and Matrix type
        """
        matrix = deepcopy(self)
        matrix -= other
        return matrix

    def __isub__(self, other: Union[float, int, bool, Matrix]) -> None:
        """
        Matrix substraction, accepts all non-str primitive types, and Matrix type
        """
        if isinstance(other, (bool, int, float)):
            for item in self._matrix:
                item += other
        
        elif isinstance(other, Matrix):
            # Now do a matrix add, we will implement two options
            # Option 1: the other Matrix row and columns are non named
            if all(name is None for name in other._row_names):
                if not all(name is None for name in other._col_names):
                    raise ValueError("For index based matrix substraction, both row and column names must be undefined (None)")

                if self._row_count != other._row_count or self._col_count != other._col_count:
                    raise ValueError("Only matrixes of equal row and column size can be substracted based on their index")

                for i in range(0, len(self._matrix)):
                    self._matrix[i] -= other._matrix[i]
                
                return self

            # Option 2: the Matrix row and columns are all named and will be added based on name
            else:
                if any(name is None for name in other._row_names) or any(name is None for name in other._col_names):
                    raise ValueError("For name based substraction, all row and column names of the subtracted matrix must be defined (NOT None)")

                for i, row_name in enumerate(other._row_names):
                    # Find index of row name
                    try:
                        index_row = self._row_names.index(row_name)
                    except ValueError:
                        continue

                    # Find index of column name
                    for j, col_name in enumerate(other._col_names):
                        try:
                            index_col = self._col_names.index(col_name)
                        except ValueError:
                            continue

                        index_self = (index_row * self._col_count) + index_col
                        index_other = (i * other._col_count) + j
                        self._matrix[index_self] -= other._matrix[index_other]

                return self
        else:
            raise ValueError("Can only substract objects of the following type to a matrix: bool, int, float, Matrix")

    def __mut__(self, other):
        raise NotImplementedError("If you really need this, ask me to implement it")

    def __imut__(self, other):
        raise NotImplementedError("If you really need this, ask me to implement it")

class CompensationMatrix:
    """
    A Compensation Matrix class, wraps a Matrix class
        :param n: the amount of fluorophore entrees
    """
    def __init__(self, n: int=5):
        self._matrix = Matrix(n, n, 0.0)

        self.__set_self_interaction()        

    @property
    def names(self) -> List[str]:
        return self._matrix.row_names

    @names.setter
    def names(self, value: List[str]) -> None:
        """
        Setter for the row and column names
        """
        self._matrix.row_names = value
        self._matrix.col_names = value

    @property
    def row_count(self) -> int:
        return self._matrix._row_count
    
    @property
    def col_count(self) -> int:
        return self._matrix._col_count

    def row(self, identifier: Union[int, str]) -> Matrix:
        return self._matrix.row(identifier)

    def col(self, identifier: Union[int, str]) -> Matrix:
        return self._matrix.col(identifier)

    def emplace_row(self, identifier: Union[int, str], data: list) -> None:
        self._matrix.emplace_row(identifier, [float(x) for x in data])
        self.__set_self_interaction()

    def append_row(self, identifier: Union[str], data: list) -> None:
        if identifier is None:
            raise ValueError("identifier should be of type str")
        
        self._matrix.append_row(identifier, [float(x) for x in data])
        #self._matrix.append_col(identifier, [5.0]*self._matrix._row_count)
        #self.__set_self_interaction()

    def erase_row(self, identifier: Union[int, str]) -> None:
        self._matrix.erase_row(identifier)
        self._matrix.erase_col(identifier)

    def emplace_col(self, identifier: Union[int, str], data: list) -> None:
        self._matrix.emplace_col(identifier, [float(x) for x in data])
        self.__set_self_interaction()

    def append_col(self, identifier: Union[str], data: list) -> None:
        if identifier is None:
            raise ValueError("identifier should be of type str")

        self._matrix.append_col(identifier, [float(x) for x in data])
        self._matrix.append_col(identifier, [0.0]*self._matrix._row_count)
        self.__set_self_interaction()

    def erase_col(self, identifier: Union[int, str]) -> None:
        self._matrix.erase_row(identifier)
        self._matrix.erase_col(identifier)

    def emplace(self, identifier: Union[int, str], row_data: list, col_data: list) -> None:
        self._matrix.emplace_row(identifier, row_data)
        self._matrix.emplace_col(identifier, col_data)
        self.__set_self_interaction()
    
    def append(self, identifier: Union[str], row_data: list, col_data: list) -> None:
        if identifier is None:
            raise ValueError("identifier should be of type str")

        self._matrix.append_row(identifier, [float(x) for x in row_data])
        col_data = [float(x) for x in col_data]
        col_data.append(1.0)
        self._matrix.append_col(identifier, col_data)
    
    def erase(self, identifier: Union[int, str]) -> None:
        self._matrix.erase_row(identifier)
        self._matrix.erase_col(identifier)
    
    def __set_self_interaction(self):
        """
        Sets the matrix interaction between identical entrees to 1.0, which is always the diagonal
        """
        for i in range(0, self._matrix._row_count):
            index = (i * self._matrix._col_count) + i
            self._matrix._matrix[index] = 1.0

    def __repr__(self):
        return self._matrix.__repr__()

    def __str__(self):
        # make a nice output
        output = ""

        # Calculate proper '\t' padding
        max_length = 0
        for name in self._matrix._row_names:
            if len(name) >= max_length:
                max_length = len(name)
        max_length += 1

        for i, name in enumerate(self._matrix._row_names):
            padding = max_length - len(name)
            output += f"\n{str(name)}:{' '*padding}"

            for value in self._matrix[i]._matrix:
                output += f"{(value*100): >6.1f}"
        
        return output[1:]

    def __add__(self, other: Union[float, int, bool, Matrix]) -> Matrix:
        item = self._matrix + other._matrix
        item.__set_self_interaction()
        return item

    def __iadd__(self, other: Union[float, int, bool, Matrix]) -> None:
        self._matrix += other._matrix
        self.__set_self_interaction()
        return self

    def __sub__(self, other: Union[float, int, bool, Matrix]) -> Matrix:
        item = self._matrix - other._matrix
        item.__set_self_interaction()
        return item

    def __isub__(self, other: Union[float, int, bool, Matrix]) -> None:
        self._matrix += other._matrix
        self.__set_self_interaction()
        return self

class MTX():
    """
    Class for opening and manipulation of FlowJo mtx files. 
    Extends a CompensationMatrix class (use .matrix attribute)
    """
    def __init__(self, path: str=None) -> None:
        self._path: str = None

        # the matrix
        self.matrix: CompensationMatrix = None

        # .mtx file properties
        self.id: str = ""
        self.name: str = ""
        self.version: str = ""
        self.status: str = ""
        self.spectral: str = ""
        self.prefix: str = ""
        self.suffix: str = ""
        self.editable: str = ""
        self.color: str = ""

        # .mtx parameter properties
        self.parameters: Dict[str, str] = {} 

        if path:
            self.path = path

    @classmethod
    def from_mtx(cls, path: str):
        """
        Alternative instantation method of the MTX class.
        As alternative to the from_wsp classmethod
        """
        return cls(path)

    @classmethod
    def from_wsp(cls, element: etree._Element):
        """
        Instantiates the matrix class from the xml tree directly extracted from a .wsp file
            :param element: the transforms:spilloverMatrix element
        """
        temp = cls(None)
        temp._parse_from_wsp(element)
        return temp

    @property
    def path(self) -> str:
        """
        Getter for path
        """
        return self._path

    @path.setter
    def path(self, path: str) -> str:
        """
        Setter for path
        """
        # Check if file exists
        if not os.path.isfile(self.path):
            raise ValueError(f"path '{path}' is invalid, path doesnt refer to a file")

        # Check the file extension (must be csv)
        _, extension = os.path.splitext(self.path)
        if extension != ".mtx":
            raise ValueError(f"path '{path}' doesnt refer to a mtx file.")

        # set file path
        self._path = path

        # Parse file and extract properties and compensation matrix
        self._parse_from_path()

    def _parse_from_wsp(self, element: etree._Element) -> None:
        """
        Extra the compensation matrix from the xml tree directly extracted from a .wsp file
            :param element: the transforms:spilloverMatrix element
        """
        matrix_data: Matrix = None

        if element.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}spilloverMatrix":
            # Get meta data
            self.spectral = element.attrib["spectral"]
            self.prefix = element.attrib["prefix"]
            self.name = element.attrib["name"]
            self.editable = element.attrib["editable"]
            self.color = element.attrib["color"]
            self.version = element.attrib["version"]
            self.status = element.attrib["status"]
            self.id = element.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}id"]
            self.suffix = element.attrib["suffix"]

            for data in element:
                # Get parameter data
                if data.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}parameters":
                    for item in data:
                        self.parameters[item.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}name"]] = item.attrib["userProvidedCompInfix"]
                
                # Get matrix data
                if data.tag == "{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}spillover":
                    # If no data ignore
                    if len(data) == 0:
                        continue
                    
                    # Else fill matrix
                    matrix = Matrix(nrow=1, ncol=0, default=0.0)
                    matrix.row_names = [data.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}parameter"]]
                    
                    for item in data:
                        matrix.append_col(item.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}parameter"], [float(item.attrib["{http://www.isac-net.org/std/Gating-ML/v2.0/transformations}value"])])

                    # Append row to matrix
                    if not matrix_data:
                        matrix_data = matrix
                    else:
                        matrix_data.append_matrix_row(matrix)
        else:
            raise ValueError("element doesnt contain a spillovermatrix")     

        # Totally not hacky upgrade to CompensationMatrix
        self.matrix = CompensationMatrix(n=matrix_data.row_count)
        self.matrix._matrix = matrix_data

    def _parse_from_path(self) -> None:
        """
        Parses the mtx file
            :raises ValueError: if the file cannot be parsed
            :returns: the compensation matrix
        """
        with open(self.path, mode='rb') as xml_file:
            xml_string = xml_file.read()

        if not xml_string:
            raise ValueError(f"path '{self.path}' refers to an empty file")

        # namespaces prefixes are non well-formed, so need to parse with recover on
        try:
            parser = etree.XMLParser(recover=True)
        except etree.XMLSyntaxError as error:
            raise ValueError(f"path '{self.path}' refers to unrecoverable xml file")

        tree = etree.fromstring(xml_string, parser)

        matrix_data: Matrix = None
        # parse iteratively, cannot get xpaths to work with non well-formed namespace prefixes...
        for parent in tree:
            if parent.tag == "transforms:spilloverMatrix":
                # Get meta data
                self.spectral = parent.attrib["spectral"]
                self.prefix = parent.attrib["prefix"]
                self.name = parent.attrib["name"]
                self.editable = parent.attrib["editable"]
                self.color = parent.attrib["color"]
                self.version = parent.attrib["version"]
                self.status = parent.attrib["status"]
                self.id = parent.attrib["transforms:id"]
                self.suffix = parent.attrib["suffix"]

                for data in parent:
                    # Get parameter data
                    if data.tag == "data-type:parameters":
                        for item in data:
                            self.parameters[item.attrib["data-type:name"]] = item.attrib["userProvidedCompInfix"]
                    
                    # Get matrix data
                    if data.tag == "transforms:spillover":
                        # If no data ignore
                        if len(data) == 0:
                            continue
                        
                        # Else fill matrix
                        matrix = Matrix(nrow=1, ncol=0, default=0.0)
                        matrix.row_names = [data.attrib["data-type:parameter"]]
                        
                        for item in data:
                            matrix.append_col(item.attrib["data-type:parameter"], [float(item.attrib["transforms:value"])])

                        # Append row to matrix
                        if not matrix_data:
                            matrix_data = matrix
                        else:
                            matrix_data.append_matrix_row(matrix)
        
        # Totally not hacky upgrade to CompensationMatrix
        self.matrix = CompensationMatrix(n=matrix_data.row_count)
        self.matrix._matrix = matrix_data

    def _dump(self) -> str:
        """
        Creates a (very) rough XML dump of this matrix
        """
        start = f"""<?xml version="1.0" encoding="UTF-8"?>\n  <gating:gatingML>\n    <transforms:spilloverMatrix spectral="{self.spectral}"  prefix="{self.prefix}"  name="{self.name}"  editable="{self.editable}"  color="{self.color}"  version="{self.version}"  status="{self.status}"  transforms:id="{self.id}"  suffix="{self.suffix}" >\n"""
        parameters = f"      <data-type:parameters>\n"
        for key in self.parameters:
            parameters += f"""        <data-type:parameter data-type:name="{key}"  userProvidedCompInfix="{self.parameters[key]}" />\n"""
        parameters += f"      </data-type:parameters>\n"

        spillover = ""
        for i in range(0, self.matrix.row_count):
            key = self.matrix.names[i]
            spillover += f"""      <transforms:spillover data-type:parameter="{key}"  userProvidedCompInfix="{self.parameters[key]}" >\n"""
            row = self.matrix.row(key)
            for j in range(0, row.col_count):
                key = row.col_names[j]
                spillover += f"""        <transforms:coefficient data-type:parameter="{key}"  transforms:value="{row._matrix[j]}" />\n"""
            spillover += "      </transforms:spillover>\n"

        end = f"""    </transforms:spilloverMatrix>\n  </gating:gatingML>\n
        """

        return start + parameters + spillover + end

    def save(self, path: str) -> None:
        """
        Saves the matrix as a flowjo .mtx file to path
            :raises ValueError: if file cannot be saved
        """
        if os.path.isfile(path):
            raise ValueError("file already exists")

        _, ex = os.path.splitext(path)

        with open(path, "w", encoding="utf-8") as file:
            file.write(self._dump())

    def __repr__(self) -> str:
        return self.matrix.__str__()

if __name__ == "__main__":
    # Get list of files .mtx files representing single stain compensation matrixes
    path = "singles"
    files = os.listdir(path)

    # Combine the single stain compensations
    matrix = None
    for single in files:
        single = MTX(os.path.join(path, single))

        if matrix:
            matrix += single
        else:
            matrix = single

    # Export combined matrix
    matrix.name = "combined"
    matrix.save("combined.mtx")
